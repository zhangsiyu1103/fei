import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets, models
from PIL import Image
import cv2
import json
import os
from models.googlenet import googlenet
from torchvision import models, transforms
import matplotlib.pyplot as plt


#Some code taken from rise original implementation for the evaluation metrics

# Function that opens image from disk, normalizes it and converts to tensor
read_tensor = transforms.Compose([
    lambda x: Image.open(x),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    lambda x: torch.unsqueeze(x, 0)
])


# Plots image from tensor
def tensor_imshow(inp, title=None, **kwargs):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # Mean and std for ImageNet
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp, **kwargs)
    if title is not None:
        plt.title(title)


# Given label number returns class name
def get_class_name(c):
    labels = np.loadtxt('synset_words.txt', str, delimiter='\t')
    return ' '.join(labels[c].split(',')[0].split()[1:])


# Image preprocessing function
preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Normalization for ImageNet
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])


# Sampler for pytorch loader. Given range r loader will only
# return dataset[r] instead of whole dataset.
class RangeSampler(Sampler):
    def __init__(self, r):
        self.r = r

    def __iter__(self):
        return iter(self.r)

    def __len__(self):
        return len(self.r)



def tensor_to_numpy(img,  is_image = True):
    if isinstance(img, np.ndarray):
        return img

    if len(img.shape) == 3:
        img = img.unsqueeze(0)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    denormalize = transforms.Normalize(
            mean= [-m/float(s) for m, s in zip(mean, std)],
            std= [1.0/s for s in std]
    )
    if is_image:
        img = denormalize(img)

    return np.uint8(255*np.transpose(img.cpu().detach().numpy(), (0, 2, 3, 1))).squeeze()


def numpy_to_tensor(img, is_image = True):
    device  ="cuda" if torch.cuda.is_available() else "cpu"
    if torch.is_tensor(img):
        return img

    if len(img.shape) == 3:
        img = img[None,:]

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalize = transforms.Normalize(mean = mean, std = std)
    new_tensor = torch.from_numpy(np.transpose(img, (0, 3, 1, 2))).float()
    if is_image:
        return normalize(new_tensor).to(device)
    else:
        return new_tensor.to(device)

def visualize_imgs(input_imgs, new_imgs, save_dir):
    if len(input_imgs.shape) == 3:
        input_imgs = input_imgs.unsqueeze(0)
    if len(new_imgs.shape) == 3:
        new_imgs = input_imgs.unsqueeze(0)


    img = tensor_to_numpy(input_imgs)
    #img = cv2.resize(img, (500,500))
    cv2.imwrite("{}/{}".format(save_dir, "original.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    img = tensor_to_numpy(new_imgs)
    #img = cv2.resize(img, (500,500))
    cv2.imwrite("{}/{}".format(save_dir, "new.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def visualize_attr(img, attr, folder_name, save_dir):
    save_dir = os.path.join(save_dir, str(folder_name))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    device  ="cuda" if torch.cuda.is_available() else "cpu"
    img_np = tensor_to_numpy(img)
    ref_np = np.zeros_like(img_np)
    shape = img.shape
    
    ref = numpy_to_tensor(ref_np, ).to(device)

    new_attr = attr.expand(img.shape)
    perturbated_preserved = tensor_to_numpy(torch.mul(img, new_attr) + torch.mul(ref, 1 - new_attr))
    perturbated_deleted = tensor_to_numpy(torch.mul(img, 1 - new_attr) + torch.mul(ref, new_attr))

    attr = attr.detach().cpu()
    shape = attr.shape
    attr = attr.view(shape[0],-1)
    attr -= attr.min(1, keepdim= True)[0]
    attr /= attr.max(1, keepdim= True)[0]
    attr = attr.view(*shape)

    heatmap = tensor_to_numpy(1-attr, False)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(heatmap, 0.6, img_np, 0.4, 0)

    heatmap = cv2.resize(heatmap, (500,500))
    img_np = cv2.resize(img_np, (500,500))
    ref_np = cv2.resize(ref_np, (500,500))
    overlay = cv2.resize(overlay, (500,500))
    perturbated_preserved = cv2.resize(perturbated_preserved, (500,500))
    perturbated_deleted = cv2.resize(perturbated_deleted, (500,500))
    cv2.imwrite("{}/{}".format(save_dir, "heatmap.png"), cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/{}".format(save_dir, "original.png"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/{}".format(save_dir, "ref.png"), cv2.cvtColor(ref_np*255, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/{}".format(save_dir, "overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/{}".format(save_dir, "perturbated_preserved.png"), cv2.cvtColor(perturbated_preserved, cv2.COLOR_RGB2BGR))
    cv2.imwrite("{}/{}".format(save_dir, "perturbated_deleted.png"), cv2.cvtColor(perturbated_deleted, cv2.COLOR_RGB2BGR))

def load_model(model_name):
    device  ="cuda" if torch.cuda.is_available() else "cpu"
    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "resnet18":
        model = models.resnet18(pretrained=True)
    elif model_name == "resnet34":
        model = models.resnet34(pretrained=True)
    elif model_name == "resnet50":
        model = models.resnet50(pretrained=True)
    elif model_name == "resnet101":
        model = models.resnet101(pretrained=True)
    elif model_name == "googlenet":
        model = googlenet()
        datadir = os.environ["PRETRAINED"]
        path = os.path.join(datadir, "googlenet.pth")
        model.load_state_dict(torch.load(path))
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)
    elif model_name == "vgg19":
        model = models.vgg19(pretrained=True)
    model = model.to(device)
    model.eval()
    return model

#def plot_loss_curve(, attrs_list):
#    areas = ["areas"]
#    loss = ["loss"]
#    plt.plot(areas, loss)
#    plt.xlabel('areas')
#    plt.ylabel('perturbation loss')
#    title = ('Loss vs Area for {} game').format(["reward_mode"])
#    if ["binary"]:
#        title += "(binary)"
#        fname = "{}_binary.png".format(["reward_mode"])
#    else:
#        fname = "{}.png".format(["reward_mode"])
#    plt.title(title)
#    plt.savefig("{}/{}".format(["plot_save_dir"], fname))
#
#    torch.save(attrs_list, "{}/attrs_list.pth".format(["plot_save_dir"]))

