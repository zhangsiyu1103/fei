import torch
from torch import nn
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter


def eval_single_perturbation(model, attrs, img, result):
    methods = [
        "ins_rand",
        "del_rand",
        "ins_blur",
        "del_blur",
        "ins_zero",
        "del_zero",
    ]

    for method in methods:
        score = eval_attr_single(model, img, attrs, method)
        if method in result:
            result[method].append(score)
        else:
            result[method] = [score]



def eval_attr_single(model, img, attr, method):
    # Convert attribution once
    attr_np = attr.detach().cpu().numpy()

    # Lazily build blur substrate if needed
    substrate_fn = None
    step = 224 * 8

    if method == "ins_rand":
        substrate_fn = torch.randn_like
        mode = "ins"
    elif method == "del_rand":
        substrate_fn = torch.randn_like
        mode = "del"
    elif method == "ins_zero":
        substrate_fn = torch.zeros_like
        mode = "ins"
    elif method == "del_zero":
        substrate_fn = torch.zeros_like
        mode = "del"
    elif method == "ins_blur":
        klen = 11
        ksig = 5
        kern = gkern(klen, ksig).to(img.device)
        def blur_substrate(x):
            return nn.functional.conv2d(x, kern, padding=klen // 2)
        substrate_fn = blur_substrate
        mode = "ins"
    elif method == "del_blur":
        klen = 11
        ksig = 5
        kern = gkern(klen, ksig).to(img.device)
        def blur_substrate(x):
            return nn.functional.conv2d(x, kern, padding=klen // 2)
        substrate_fn = blur_substrate
        mode = "del"
    else:
        raise ValueError(f"Unknown method: {method}")

    metric = CausalMetric(model, mode, step, substrate_fn)
    with torch.no_grad():
        scores = metric.evaluate(img, attr_np, img.shape[0])
    return auc(scores.mean(1))

def gkern(klen, nsig):
    """Returns a Gaussian kernel array."""
    grid = np.zeros((klen, klen))
    grid[klen // 2, klen // 2] = 1
    k = gaussian_filter(grid, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (len(arr) - 1)


class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.device = next(model.parameters()).device

    def single_run(self, img_tensor, explanation, verbose=0, save_to=None):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        C, H, W = img_tensor.shape[1:]
        HW = H * W
        
        # Ensure image tensor is on the correct device
        img_tensor = img_tensor.to(self.device)

        pred = self.model(img_tensor)
        top, c = torch.max(pred, 1)
        c = c.item() # Use .item() for single value tensors
        n_steps = (HW + self.step - 1) // self.step

        if self.mode == 'del':
            start = img_tensor.clone()
            finish = self.substrate_fn(img_tensor)
        elif self.mode == 'ins':
            start = self.substrate_fn(img_tensor)
            finish = img_tensor.clone()

        scores = np.empty(n_steps + 1)
        salient_order = np.flip(np.argsort(explanation.reshape(-1, HW), axis=1), axis=-1)
        
        # Flatten tensors for easier indexing
        start_flat = start.view(1, C, HW)
        finish_flat = finish.view(1, C, HW)

        for i in range(n_steps + 1):
            pred = self.model(start)
            scores[i] = pred[0, c].item()

            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start_flat[0, :, coords] = finish_flat[0, :, coords]

        print(f'AUC: {auc(scores):.4f}')
        return scores

    def evaluate(self, img_batch, exp_batch, batch_size):
        r"""Efficiently evaluate a big batch of images.
        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.
        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples, C, H, W = img_batch.shape
        HW = H * W
        n_classes = self.model(img_batch.to(self.device)).shape[1]

        img_batch = img_batch.to(self.device)
        predictions = torch.zeros(n_samples, n_classes)
        for i in tqdm(range(0, n_samples, batch_size), desc='Predicting labels'):
            preds = self.model(img_batch[i:i+batch_size]).cpu()
            predictions[i:i+batch_size] = preds
        top_class_indices = torch.argmax(predictions, dim=-1)

        salient_order = torch.from_numpy(
            np.flip(np.argsort(exp_batch.reshape(n_samples, HW), axis=1), axis=-1).copy()
        ).long().to(self.device)

        substrate = self.substrate_fn(img_batch)
        
        if self.mode == 'del':
            caption = 'Deletion'
            start = img_batch.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Insertion'
            start = substrate.clone()
            finish = img_batch

        start_flat = start.view(n_samples, C, HW)
        finish_flat = finish.view(n_samples, C, HW)

        n_steps = (HW + self.step - 1) // self.step
        scores = torch.zeros((n_steps + 1, n_samples))

        batch_indices = torch.arange(n_samples, device=self.device).view(-1, 1)

        for i in tqdm(range(n_steps + 1), desc=f'{caption} metric'):
            for j in range(0, n_samples, batch_size):
                preds = self.model(start[j:j+batch_size])
                scores[i, j:j+batch_size] = preds.gather(1, top_class_indices[j:j+batch_size].to(self.device).unsqueeze(1)).squeeze().cpu()

            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)]
                start_flat[batch_indices, :, coords] = finish_flat[batch_indices, :, coords]

        scores = scores.numpy()
        mean_auc = auc(scores.mean(axis=1))
        print(f'Mean AUC: {mean_auc:.4f}')
        return scores