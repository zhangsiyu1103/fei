import torch
import torch.nn as nn
import os
import numpy as np
import argparse
from torchvision import datasets, transforms
import utils
from cub_tools.transforms import makeDefaultTransforms
from eval_metric import eval_single_perturbation
import csv

def load_saved_attributions(save_dir, methods=None, dataset=None, model_name=None):
    """
    Load attributions saved as .npy files in save_dir/dataset/model_method.npy format.

    Args:
        save_dir (str): Top-level directory containing saved attributions
        methods (list, optional): List of methods to load. If None, loads all available.
        dataset (str, optional): Dataset name (e.g., 'imagenet', 'cub')
        model_name (str, optional): Model name (e.g., 'vgg16')

    Returns:
        dict: Dictionary with method names as keys and attributions as values (torch.Tensor)
    """
    attributions = {}
    if dataset is None or model_name is None:
        raise ValueError("Both dataset and model_name must be specified for attribution loading.")

    folder = save_dir
    if not os.path.exists(folder):
        print(f"Attribution folder does not exist: {folder}")
        return attributions

    all_files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    if methods is None:
        methods = [fname.replace('.npy', '') for fname in all_files]

    for method in methods:
        file_path = os.path.join(folder, f"{method}.npy")
        if os.path.exists(file_path):
            try:
                attrs = np.load(file_path)
                if method == "FGVIS_attributions":
                    attributions[method] = torch.from_numpy(attrs).float().mean(dim=2)
                else:
                    attributions[method] = torch.from_numpy(attrs).float().squeeze().unsqueeze(1)
                print(attributions[method].shape)
                print(f"Loaded {len(attrs)} attributions for {method} from {file_path}")
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        else:
            print(f"Attribution file not found: {file_path}")

    return attributions



def evaluate_attributions_with_metrics(model, inputs, saved_attributions, start_idx, end_idx):
    """
    Evaluate saved attributions using insertion/deletion metrics from eval_metric.py.

    Args:
        model: The model for evaluation (wrapped with Softmax)
        inputs: Input tensors for current batch
        saved_attributions: Dictionary of saved attribution methods (all images)
        start_idx: Start index in the attribution array
        end_idx: End index in the attribution array

    Returns:
        dict: Evaluation results for each attribution method and perturbation type
    """
    batch_results = {}

    # Process each attribution method
    for method_name, attributions in saved_attributions.items():
        cur_inputs = inputs
        cur_attributions = attributions[start_idx:end_idx]

        # Evaluate using perturbation metrics from eval_metric.py
        method_result = {}
        eval_single_perturbation(model, cur_attributions, cur_inputs, method_result)

        batch_results[method_name] = method_result
        print(f"Completed metric evaluation for {method_name}")

    return batch_results


def metric_eval_attributions(args):
    """
    Main function to load saved attributions and evaluate them using insertion/deletion metrics.

    Args:
        args: Arguments containing model, dataset, and save directory information
    """
    # Load dataset
    if args.dataset == "imagenet":
        datadir = os.environ.get("IMAGENETDIR")
        if datadir is None:
            raise RuntimeError("IMAGENETDIR env var not set")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        cur_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        test = datasets.ImageFolder(os.path.join(datadir, "val"), cur_transform)
    elif args.dataset == "cub":
        datadir = os.environ.get("CUBDIR")
        if datadir is None:
            raise RuntimeError("CUBDIR env var not set (path to CUB_200_2011)")
        data_transforms = makeDefaultTransforms()
        test = datasets.ImageFolder(os.path.join(datadir, 'images', 'test'), data_transforms['test'])

    # Load model
    model = utils.load_model(args.model, args.dataset)

    model.eval()

    # Wrap model with Softmax for metric evaluation
    metric_model = nn.Sequential(model, nn.Softmax(dim=1))

    # Load saved attributions
    print("Loading saved attributions...")
    saved_attributions = load_saved_attributions(args.attr_dir, dataset=args.dataset, model_name=args.model)

    if not saved_attributions:
        print("No saved attributions found!")
        return

    # Load data indices
    if args.dataset == "imagenet":
        sampled_indices = np.loadtxt("data_index_imagenet.txt", int, delimiter=",").tolist()
        sampled_subset = torch.utils.data.Subset(test, sampled_indices)
    else:
        sampled_indices = np.loadtxt("data_index_cub.txt", int, delimiter=",").tolist()
        sampled_subset = torch.utils.data.Subset(test, sampled_indices)

    test_loader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Process each batch
    all_results = {}

    for idx, (cur_input, _) in enumerate(test_loader):
        print(f"Processing batch {idx+1}/{len(test_loader)}")
        cur_input = cur_input.to(args.device)

        start_idx = args.batch_size * idx
        end_idx = start_idx + cur_input.shape[0]

        # Evaluate using insertion/deletion metrics
        print("Evaluating with insertion/deletion metrics...")
        batch_results = evaluate_attributions_with_metrics(
            metric_model, cur_input, saved_attributions, start_idx, end_idx
        )

        # Accumulate results
        for method_name, method_results in batch_results.items():
            if method_name not in all_results:
                all_results[method_name] = {}

            for perturbation_method, scores in method_results.items():
                if perturbation_method not in all_results[method_name]:
                    all_results[method_name][perturbation_method] = []

                # scores is a list containing one AUC value per image in the batch
                all_results[method_name][perturbation_method].extend(scores)

    # Save results
    print("Saving results...")
    results_file = os.path.join(args.save_dir, "metric_eval_results.pth")
    torch.save(all_results, results_file)

    # Save CSV summary
    csv_path = os.path.join(args.save_dir, "metric_eval_summary.csv")

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ["attribution_method", "perturbation_method", "mean_auc", "std_auc", "num_samples"]
        writer.writerow(header)

        for method_name, method_results in all_results.items():
            for perturbation_method, auc_values in method_results.items():
                if auc_values:
                    mean_auc = np.mean(auc_values)
                    std_auc = np.std(auc_values)
                    num_samples = len(auc_values)
                    writer.writerow([method_name, perturbation_method, mean_auc, std_auc, num_samples])

    print(f"Results saved to {results_file}")
    print(f"Summary saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser("metric_eval")
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--model', type=str, default='vgg16',
                       choices=["vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "googlenet", "alexnet"],
                       help='model architecture')
    parser.add_argument('--dataset', type=str, default='imagenet',
                       choices=["imagenet", "cub"], help='dataset to work on')
    parser.add_argument('--save_dir', type=str, default='./result_faithful', help='directory to save results')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--attr_dir', type=str, default='./result_save', help='directory with saved attributions')

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_dir = f"{args.save_dir}/{args.dataset}/{args.model}"
    args.attr_dir = f"{args.attr_dir}/{args.dataset}/{args.model}"
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Starting metric evaluation for {args.model} on {args.dataset}")
    print(f"Save directory: {args.save_dir}")

    metric_eval_attributions(args)


if __name__ == '__main__':
    main()
