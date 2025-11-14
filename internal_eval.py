import torch
import torch.nn as nn
import os
import numpy as np
import argparse
from wrapper import Wrapper
from torchvision import datasets, transforms
from cub_tools.transforms import makeDefaultTransforms
from eval_metric import gkern
import csv
import utils

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
    print(methods)
    for method in methods:
        file_path = os.path.join(folder, f"{method}.npy")
        if os.path.exists(file_path):
            try:
                attrs = np.load(file_path)
                if method == "FGVIS_attributions":
                    attributions[method] = torch.from_numpy(attrs).float().mean(dim=2)
                else:
                    attributions[method] = torch.from_numpy(attrs).float().squeeze().unsqueeze(1)
                print(f"Loaded {len(attrs)} attributions for {method} from {file_path}")
            except Exception as e:
                print(f"Failed to load {file_path}: {e}")
        else:
            print(f"Attribution file not found: {file_path}")

    return attributions


def get_intermediate_relu_layers(model, model_name):
    """
    Get intermediate ReLU layers from the wrapped model.

    Args:
        model: Original model
        model_name (str): Name of the model architecture

    Returns:
        list: List of (layer, name) tuples for intermediate ReLU layers
    """
    # Get the actual layer objects
    if model_name == "resnet50":
        target_layers = [model.layer1[-1], model.layer2[-1], model.layer3[-1], model.layer4[-1]]
        layer_names = ["layer1[-1]", "layer2[-1]", "layer3[-1]", "layer4[-1]"]
    elif model_name == "vgg16":
        target_layers = [model.features[3], model.features[8], model.features[15], model.features[22], model.features[29]]
        layer_names = ["features[3]", "features[8]", "features[15]", "features[22]", "features[29]"]
    else:
        raise ValueError(f"Model {model_name} not supported")
    print(target_layers)
    # Return layers directly with their names
    intermediate_layers = [(layer, name) for layer, name in zip(target_layers, layer_names)]
    return intermediate_layers


def get_perturbation_config(method, device):
    """
    Get substrate function and mode for a perturbation method.
    Extracted outside of loops to avoid redundant definitions.

    Args:
        method: Perturbation method name
        device: torch device

    Returns:
        tuple: (substrate_fn, mode)
    """
    if method == "ins_rand":
        return lambda x: torch.randn_like(x), "ins"
    elif method == "del_rand":
        return lambda x: torch.randn_like(x), "del"
    elif method == "ins_zero":
        return lambda x: torch.zeros_like(x), "ins"
    elif method == "del_zero":
        return lambda x: torch.zeros_like(x), "del"
    elif method == "ins_blur":
        klen, ksig = 11, 5
        kern = gkern(klen, ksig).to(device)
        return lambda x: nn.functional.conv2d(x, kern, padding=klen//2), "ins"
    elif method == "del_blur":
        klen, ksig = 11, 5
        kern = gkern(klen, ksig).to(device)
        return lambda x: nn.functional.conv2d(x, kern, padding=klen//2), "del"
    else:
        raise ValueError(f"Unknown perturbation method: {method}")


def compute_unit_overlap(model, inputs, attributions, intermediate_layers, perturbation_methods, step=224*8, threshold=0.0):
    """
    Compute overlap of activated and inactivated units between original and perturbed images.
    Memory-optimized version that computes metrics directly in hooks.
    Returns per-step values for AUC computation.

    Args:
        model: The model to compute activations for
        inputs: Input tensors (N, C, H, W)
        attributions: Attribution maps for perturbation ordering (N, C, H, W)
        intermediate_layers: List of (layer, name) tuples for intermediate ReLU layers
        perturbation_methods: List of perturbation methods
        step: Number of pixels to modify per step
        threshold: Threshold to determine if a unit is activated (default: 0.0)

    Returns:
        dict: Dictionary with {method: {layer_name: {'activated_stay_active_steps': [...], 'inactivated_stay_inactive_steps': [...]}}}
              Each list contains sublists per perturbation step, where each sublist has values per sample
    """
    device = next(model.parameters()).device
    n_samples, C, H, W = inputs.shape
    HW = H * W

    results = {}

    # Compute original activations once for all layers (store as boolean masks to save memory)
    original_activated = {}

    def make_orig_hook(layer_name):
        def hook(module, inp, output):
            output_flat = output.detach().view(n_samples, -1)
            original_activated[layer_name] = (output_flat > threshold)
        return hook

    hooks = []
    for layer, name in intermediate_layers:
        hooks.append(layer.register_forward_hook(make_orig_hook(name)))

    with torch.no_grad():
        _ = model(inputs)

    for h in hooks:
        h.remove()

    # Process each perturbation method
    for method in perturbation_methods:
        results[method] = {name: {'activated_stay_active_steps': [], 'inactivated_stay_inactive_steps': []}
                          for _, name in intermediate_layers}

        # Get substrate function and mode (extracted outside loop)
        substrate_fn, mode = get_perturbation_config(method, device)

        # Compute pixel ordering for all samples (N, HW)
        flat_attr = attributions.view(n_samples, -1)
        if mode == "del":
            order = torch.argsort(flat_attr, dim=1, descending=True)
        else:
            order = torch.argsort(flat_attr, dim=1, descending=False)

        # Precompute substrate once
        substrate = substrate_fn(inputs)

        # Process perturbation steps
        num_steps = (HW + step - 1) // step

        for s in range(num_steps + 1):
            # Build mask for current perturbation level
            if s == 0:
                # No perturbation - use original images
                pert_imgs = inputs
            else:
                # Determine pixels to perturb up to step s
                end_idx = min(s * step, HW)
                pixel_indices = order[:, :end_idx]  # (N, end_idx)

                # Create mask efficiently
                mask = torch.ones(n_samples, HW, device=device)
                batch_idx = torch.arange(n_samples, device=device).unsqueeze(1)
                mask[batch_idx, pixel_indices] = 0
                mask = mask.view(n_samples, 1, H, W)

                # Apply perturbation
                pert_imgs = inputs * mask + substrate * (1 - mask)

            # Storage for current step
            step_activated_ratios = {name: [] for _, name in intermediate_layers}
            step_inactivated_ratios = {name: [] for _, name in intermediate_layers}

            # Compute metrics directly in hooks (function defined once outside loop)
            def make_pert_hook(layer_name):
                def hook(module, inp, output):
                    output_flat = output.detach().view(n_samples, -1)
                    pert_activated = (output_flat > threshold)

                    # Compute ratio of originally-activated neurons that stay active
                    orig_act = original_activated[layer_name]
                    orig_act_count = orig_act.sum(dim=1)
                    stay_active_count = (orig_act & pert_activated).sum(dim=1)
                    activated_ratio = (stay_active_count / (orig_act_count + 1e-8)).tolist()

                    # Compute ratio of originally-inactivated neurons that stay inactive
                    orig_inact = torch.logical_not(orig_act)
                    orig_inact_count = orig_inact.sum(dim=1)
                    stay_inactive_count = (orig_inact & torch.logical_not(pert_activated)).sum(dim=1)
                    inactivated_ratio = (stay_inactive_count / (orig_inact_count + 1e-8)).tolist()

                    step_activated_ratios[layer_name] = activated_ratio
                    step_inactivated_ratios[layer_name] = inactivated_ratio
                return hook

            pert_hooks = []
            for layer, name in intermediate_layers:
                pert_hooks.append(layer.register_forward_hook(make_pert_hook(name)))

            with torch.no_grad():
                _ = model(pert_imgs)

            for h in pert_hooks:
                h.remove()

            # Store step results
            for _, name in intermediate_layers:
                results[method][name]['activated_stay_active_steps'].append(step_activated_ratios[name])
                results[method][name]['inactivated_stay_inactive_steps'].append(step_inactivated_ratios[name])

    return results


def compute_mse_activation_loss(model, inputs, attributions, intermediate_layers, perturbation_methods, step=224*8):
    """
    Memory-optimized computation of MSE loss between intermediate activations of perturbed and original images.
    Computes MSE directly in hooks to avoid storing full activation tensors.
    Returns per-step values for averaging.

    Args:
        model: The model to compute activations for
        inputs: Input tensors (N, C, H, W)
        attributions: Attribution maps for perturbation ordering (N, C, H, W)
        intermediate_layers: List of (layer, name) tuples for intermediate ReLU layers
        perturbation_methods: List of perturbation methods
        step: Number of pixels to modify per step

    Returns:
        dict: Dictionary with {method: {layer_name: {'mse_steps': [...]}}}
              Each list contains sublists per perturbation step, where each sublist has MSE values per sample
    """
    device = next(model.parameters()).device
    n_samples, C, H, W = inputs.shape
    HW = H * W

    results = {}

    # Compute original activations once for all layers
    original_acts = {}

    def make_orig_hook(layer_name):
        def hook(module, inp, output):
            original_acts[layer_name] = output.detach()
        return hook

    hooks = []
    for layer, name in intermediate_layers:
        hooks.append(layer.register_forward_hook(make_orig_hook(name)))

    with torch.no_grad():
        _ = model(inputs)

    for h in hooks:
        h.remove()

    # Process each perturbation method
    for method in perturbation_methods:
        results[method] = {name: {'mse_steps': []} for _, name in intermediate_layers}

        # Get substrate function and mode (extracted outside loop)
        substrate_fn, mode = get_perturbation_config(method, device)

        # Compute pixel ordering for all samples (N, HW)
        flat_attr = attributions.view(n_samples, -1)
        if mode == "del":
            order = torch.argsort(flat_attr, dim=1, descending=True)
        else:
            order = torch.argsort(flat_attr, dim=1, descending=False)

        # Precompute substrate once
        substrate = substrate_fn(inputs)

        # Process perturbation steps
        num_steps = (HW + step - 1) // step

        for s in range(num_steps + 1):
            # Build mask for current perturbation level
            if s == 0:
                # No perturbation - use original images
                pert_imgs = inputs
            else:
                # Determine pixels to perturb up to step s
                end_idx = min(s * step, HW)
                pixel_indices = order[:, :end_idx]  # (N, end_idx)

                # Create mask efficiently
                mask = torch.ones(n_samples, HW, device=device)
                batch_idx = torch.arange(n_samples, device=device).unsqueeze(1)
                mask[batch_idx, pixel_indices] = 0
                mask = mask.view(n_samples, 1, H, W)

                # Apply perturbation
                pert_imgs = inputs * mask + substrate * (1 - mask)

            # Storage for current step
            step_mse_values = {name: [] for _, name in intermediate_layers}

            # Compute MSE directly in hooks (function defined once outside loop)
            def make_pert_hook(layer_name):
                def hook(module, inp, output):
                    orig = original_acts[layer_name]
                    pert = output.detach()
                    # Compute MSE per sample, flattening spatial/channel dimensions
                    mse_per_sample = ((orig - pert).view(n_samples, -1) ** 2).mean(dim=1)
                    step_mse_values[layer_name] = mse_per_sample.cpu().tolist()
                return hook

            pert_hooks = []
            for layer, name in intermediate_layers:
                pert_hooks.append(layer.register_forward_hook(make_pert_hook(name)))

            with torch.no_grad():
                _ = model(pert_imgs)

            for h in pert_hooks:
                h.remove()

            # Store step results
            for _, name in intermediate_layers:
                results[method][name]['mse_steps'].append(step_mse_values[name])

    return results


def compute_correlation(model, inputs, attributions, intermediate_layers, perturbation_methods, step=224 * 8):
    """
    Memory-optimized computation of correlation between intermediate activations of perturbed and original images.
    Computes correlation directly in hooks to avoid storing full activation tensors.
    Returns per-step values for AUC computation.

    Args:
        model: The model to compute activations for
        inputs: Input tensors (N, C, H, W)
        attributions: Attribution maps for perturbation ordering (N, C, H, W)
        intermediate_layers: List of (layer, name) tuples for intermediate ReLU layers
        perturbation_methods: List of perturbation methods
        step: Number of pixels to modify per step

    Returns:
        dict: Dictionary with {method: {layer_name: {'correlation_steps': [...]}}}
              Each list contains sublists per perturbation step, where each sublist has correlation values per sample
    """
    device = next(model.parameters()).device
    n_samples, C, H, W = inputs.shape
    HW = H * W

    results = {}

    # Compute original activations once for all layers
    original_acts = {}

    def make_orig_hook(layer_name):
        def hook(module, inp, output):
            original_acts[layer_name] = output.detach()
        return hook

    hooks = []
    for layer, name in intermediate_layers:
        hooks.append(layer.register_forward_hook(make_orig_hook(name)))

    with torch.no_grad():
        _ = model(inputs)

    for h in hooks:
        h.remove()

    # Process each perturbation method
    for method in perturbation_methods:
        results[method] = {name: {'correlation_steps': []} for _, name in intermediate_layers}

        # Get substrate function and mode (extracted outside loop)
        substrate_fn, mode = get_perturbation_config(method, device)

        # Compute pixel ordering for all samples (N, HW)
        flat_attr = attributions.view(n_samples, -1)
        if mode == "del":
            order = torch.argsort(flat_attr, dim=1, descending=True)
        else:
            order = torch.argsort(flat_attr, dim=1, descending=False)

        # Precompute substrate once
        substrate = substrate_fn(inputs)

        # Process perturbation steps
        num_steps = (HW + step - 1) // step

        for s in range(num_steps + 1):
            # Build mask for current perturbation level
            if s == 0:
                # No perturbation - use original images
                pert_imgs = inputs
            else:
                # Determine pixels to perturb up to step s
                end_idx = min(s * step, HW)
                pixel_indices = order[:, :end_idx]  # (N, end_idx)

                # Create mask efficiently
                mask = torch.ones(n_samples, HW, device=device)
                batch_idx = torch.arange(n_samples, device=device).unsqueeze(1)
                mask[batch_idx, pixel_indices] = 0
                mask = mask.view(n_samples, 1, H, W)

                # Apply perturbation
                pert_imgs = inputs * mask + substrate * (1 - mask)

            # Storage for current step
            step_corr_values = {name: [] for _, name in intermediate_layers}

            # Compute correlation directly in hooks (vectorized)
            def make_pert_hook(layer_name):
                def hook(module, inp, output):
                    orig = original_acts[layer_name]
                    pert = output.detach()
                    # Flatten spatial/channel dimensions: (N, -1)
                    orig_flat = orig.view(n_samples, -1)
                    pert_flat = pert.view(n_samples, -1)

                    # Vectorized correlation computation
                    # Center the vectors along feature dimension
                    orig_centered = orig_flat - orig_flat.mean(dim=1, keepdim=True)  # (N, D)
                    pert_centered = pert_flat - pert_flat.mean(dim=1, keepdim=True)  # (N, D)

                    # Compute correlation for all samples at once
                    numerator = (orig_centered * pert_centered).sum(dim=1)  # (N,)
                    denominator = torch.sqrt((orig_centered ** 2).sum(dim=1) * (pert_centered ** 2).sum(dim=1))  # (N,)

                    # Handle zero denominators
                    corr = torch.where(denominator > 1e-8, numerator / denominator, torch.zeros_like(numerator))

                    step_corr_values[layer_name] = corr.cpu().tolist()
                return hook

            pert_hooks = []
            for layer, name in intermediate_layers:
                pert_hooks.append(layer.register_forward_hook(make_pert_hook(name)))

            with torch.no_grad():
                _ = model(pert_imgs)

            for h in pert_hooks:
                h.remove()

            # Store step results
            for _, name in intermediate_layers:
                results[method][name]['correlation_steps'].append(step_corr_values[name])

    return results


def compute_cosine_similarity(model, inputs, attributions, intermediate_layers, perturbation_methods, step=224 * 8):
    """
    Memory-optimized computation of cosine similarity between intermediate activations of perturbed and original images.
    Computes cosine similarity directly in hooks to avoid storing full activation tensors.
    Returns per-step values for mean computation.

    Args:
        model: The model to compute activations for
        inputs: Input tensors (N, C, H, W)
        attributions: Attribution maps for perturbation ordering (N, C, H, W)
        intermediate_layers: List of (layer, name) tuples for intermediate ReLU layers
        perturbation_methods: List of perturbation methods
        step: Number of pixels to modify per step

    Returns:
        dict: Dictionary with {method: {layer_name: {'cosine_steps': [...]}}}
              Each list contains sublists per perturbation step, where each sublist has cosine similarity values per sample
    """
    device = next(model.parameters()).device
    n_samples, C, H, W = inputs.shape
    HW = H * W

    results = {}

    # Compute original activations once for all layers
    original_acts = {}

    def make_orig_hook(layer_name):
        def hook(module, inp, output):
            original_acts[layer_name] = output.detach()
        return hook

    hooks = []
    for layer, name in intermediate_layers:
        hooks.append(layer.register_forward_hook(make_orig_hook(name)))

    with torch.no_grad():
        _ = model(inputs)

    for h in hooks:
        h.remove()

    # Process each perturbation method
    for method in perturbation_methods:
        results[method] = {name: {'cosine_steps': []} for _, name in intermediate_layers}

        # Get substrate function and mode (extracted outside loop)
        substrate_fn, mode = get_perturbation_config(method, device)

        # Compute pixel ordering for all samples (N, HW)
        flat_attr = attributions.view(n_samples, -1)
        if mode == "del":
            order = torch.argsort(flat_attr, dim=1, descending=True)
        else:
            order = torch.argsort(flat_attr, dim=1, descending=False)

        # Precompute substrate once
        substrate = substrate_fn(inputs)

        # Process perturbation steps
        num_steps = (HW + step - 1) // step

        for s in range(num_steps + 1):
            # Build mask for current perturbation level
            if s == 0:
                # No perturbation - use original images
                pert_imgs = inputs
            else:
                # Determine pixels to perturb up to step s
                end_idx = min(s * step, HW)
                pixel_indices = order[:, :end_idx]  # (N, end_idx)

                # Create mask efficiently
                mask = torch.ones(n_samples, HW, device=device)
                batch_idx = torch.arange(n_samples, device=device).unsqueeze(1)
                mask[batch_idx, pixel_indices] = 0
                mask = mask.view(n_samples, 1, H, W)

                # Apply perturbation
                pert_imgs = inputs * mask + substrate * (1 - mask)

            # Storage for current step
            step_cosine_values = {name: [] for _, name in intermediate_layers}

            # Compute cosine similarity directly in hooks (vectorized)
            def make_pert_hook(layer_name):
                def hook(module, inp, output):
                    orig = original_acts[layer_name]
                    pert = output.detach()
                    # Flatten spatial/channel dimensions: (N, -1)
                    orig_flat = orig.view(n_samples, -1)
                    pert_flat = pert.view(n_samples, -1)

                    # Vectorized cosine similarity computation
                    # Compute dot product for all samples at once
                    numerator = (orig_flat * pert_flat).sum(dim=1)  # (N,)

                    # Compute norms for all samples at once
                    orig_norm = torch.sqrt((orig_flat ** 2).sum(dim=1))  # (N,)
                    pert_norm = torch.sqrt((pert_flat ** 2).sum(dim=1))  # (N,)
                    denominator = orig_norm * pert_norm  # (N,)

                    # Handle zero denominators
                    cosine = torch.where(denominator > 1e-8, numerator / denominator, torch.zeros_like(numerator))

                    step_cosine_values[layer_name] = cosine.cpu().tolist()
                return hook

            pert_hooks = []
            for layer, name in intermediate_layers:
                pert_hooks.append(layer.register_forward_hook(make_pert_hook(name)))

            with torch.no_grad():
                _ = model(pert_imgs)

            for h in pert_hooks:
                h.remove()

            # Store step results
            for _, name in intermediate_layers:
                results[method][name]['cosine_steps'].append(step_cosine_values[name])

    return results


def compute_auc_trapezoidal(values_per_step, num_samples):
    """
    Compute AUC using trapezoidal integration for per-step values.

    Args:
        values_per_step: List of lists, where each inner list contains values for all samples at that step
        num_samples: Number of samples

    Returns:
        list: AUC value for each sample
    """
    num_steps = len(values_per_step)
    if num_steps == 0:
        return [0.0] * num_samples

    # Convert to array: (num_steps, num_samples)
    values_array = np.array(values_per_step)

    # x-axis: normalized perturbation level (0 to 1)
    x = np.linspace(0, 1, num_steps)

    # Compute AUC for each sample using trapezoidal rule
    aucs = []
    for sample_idx in range(num_samples):
        y = values_array[:, sample_idx]
        auc = np.trapz(y, x)
        aucs.append(auc)

    return aucs


def evaluate_intermediate_activations_mse(model, inputs, saved_attributions, intermediate_layers, start_idx, end_idx):
    """
    Evaluate intermediate activations using MSE loss between perturbed and original activations.

    Args:
        model: The model for evaluation
        inputs: Input tensors for current batch
        saved_attributions: Dictionary of saved attribution methods (all images)
        intermediate_layers: List of intermediate ReLU layers
        start_idx: Start index in the attribution array
        end_idx: End index in the attribution array

    Returns:
        dict: Evaluation results with average MSE across steps
    """
    perturbation_methods = ["ins_rand", "del_rand", "ins_blur", "del_blur", "ins_zero", "del_zero"]
    batch_results = {}

    # Process each attribution method
    for method_name, attributions in saved_attributions.items():
        cur_inputs = inputs
        cur_attributions = attributions[start_idx:end_idx]

        # Compute MSE activation loss
        method_results = compute_mse_activation_loss(
            model, cur_inputs, cur_attributions, intermediate_layers, perturbation_methods
        )

        # Compute average MSE across steps for each sample
        batch_results[method_name] = {}
        for perturbation_method, layer_results in method_results.items():
            batch_results[method_name][perturbation_method] = {}
            for layer_name, mse_dict in layer_results.items():
                mse_steps = mse_dict['mse_steps']
                # Average across steps for each sample
                mse_array = np.array(mse_steps)  # (num_steps, num_samples)
                avg_mse_per_sample = mse_array.mean(axis=0).tolist()
                batch_results[method_name][perturbation_method][layer_name] = avg_mse_per_sample

        print(f"Completed MSE evaluation for {method_name}")

    return batch_results


def evaluate_unit_overlap(model, inputs, saved_attributions, intermediate_layers, start_idx, end_idx):
    """
    Evaluate unit overlap (activated and inactivated) between perturbed and original activations.
    Computes AUC using trapezoidal integration.

    Args:
        model: The model for evaluation
        inputs: Input tensors for current batch
        saved_attributions: Dictionary of saved attribution methods (all images)
        intermediate_layers: List of intermediate ReLU layers
        start_idx: Start index in the attribution array
        end_idx: End index in the attribution array

    Returns:
        dict: Evaluation results with AUC for overlap metrics
    """
    perturbation_methods = ["ins_rand", "del_rand", "ins_blur", "del_blur", "ins_zero", "del_zero"]
    batch_results = {}
    n_samples = inputs.shape[0]

    # Process each attribution method
    for method_name, attributions in saved_attributions.items():
        cur_inputs = inputs
        cur_attributions = attributions[start_idx:end_idx]

        # Compute unit overlap
        method_results = compute_unit_overlap(
            model, cur_inputs, cur_attributions, intermediate_layers, perturbation_methods
        )

        # Compute AUC for each sample
        batch_results[method_name] = {}
        for perturbation_method, layer_results in method_results.items():
            batch_results[method_name][perturbation_method] = {}
            for layer_name, overlap_dict in layer_results.items():
                activated_steps = overlap_dict['activated_stay_active_steps']
                inactivated_steps = overlap_dict['inactivated_stay_inactive_steps']

                # Compute AUC using trapezoidal integration
                activated_auc = compute_auc_trapezoidal(activated_steps, n_samples)
                inactivated_auc = compute_auc_trapezoidal(inactivated_steps, n_samples)

                batch_results[method_name][perturbation_method][layer_name] = {
                    'activated_stay_active_auc': activated_auc,
                    'inactivated_stay_inactive_auc': inactivated_auc
                }

        print(f"Completed unit overlap evaluation for {method_name}")

    return batch_results


def evaluate_correlation(model, inputs, saved_attributions, intermediate_layers, start_idx, end_idx):
    """
    Evaluate correlation between perturbed and original activations.
    Computes AUC using trapezoidal integration.

    Args:
        model: The model for evaluation
        inputs: Input tensors for current batch
        saved_attributions: Dictionary of saved attribution methods (all images)
        intermediate_layers: List of intermediate ReLU layers
        start_idx: Start index in the attribution array
        end_idx: End index in the attribution array

    Returns:
        dict: Evaluation results with AUC for correlation metrics
    """
    perturbation_methods = ["ins_rand", "del_rand", "ins_blur", "del_blur", "ins_zero", "del_zero"]
    batch_results = {}
    n_samples = inputs.shape[0]

    # Process each attribution method
    for method_name, attributions in saved_attributions.items():
        cur_inputs = inputs
        cur_attributions = attributions[start_idx:end_idx]

        # Compute correlation
        method_results = compute_correlation(
            model, cur_inputs, cur_attributions, intermediate_layers, perturbation_methods
        )

        # Compute AUC for each sample
        batch_results[method_name] = {}
        for perturbation_method, layer_results in method_results.items():
            batch_results[method_name][perturbation_method] = {}
            for layer_name, corr_dict in layer_results.items():
                correlation_steps = corr_dict['correlation_steps']

                # Compute AUC using trapezoidal integration
                correlation_auc = compute_auc_trapezoidal(correlation_steps, n_samples)

                batch_results[method_name][perturbation_method][layer_name] = correlation_auc

        print(f"Completed correlation evaluation for {method_name}")

    return batch_results


def evaluate_cosine_similarity(model, inputs, saved_attributions, intermediate_layers, start_idx, end_idx):
    """
    Evaluate cosine similarity between perturbed and original activations.
    Computes mean across steps.

    Args:
        model: The model for evaluation
        inputs: Input tensors for current batch
        saved_attributions: Dictionary of saved attribution methods (all images)
        intermediate_layers: List of intermediate ReLU layers
        start_idx: Start index in the attribution array
        end_idx: End index in the attribution array

    Returns:
        dict: Evaluation results with mean for cosine similarity metrics
    """
    perturbation_methods = ["ins_rand", "del_rand", "ins_blur", "del_blur", "ins_zero", "del_zero"]
    batch_results = {}
    n_samples = inputs.shape[0]

    # Process each attribution method
    for method_name, attributions in saved_attributions.items():
        cur_inputs = inputs
        cur_attributions = attributions[start_idx:end_idx]

        # Compute cosine similarity
        method_results = compute_cosine_similarity(
            model, cur_inputs, cur_attributions, intermediate_layers, perturbation_methods
        )

        # Compute mean across steps for each sample
        batch_results[method_name] = {}
        for perturbation_method, layer_results in method_results.items():
            batch_results[method_name][perturbation_method] = {}
            for layer_name, cosine_dict in layer_results.items():
                cosine_steps = cosine_dict['cosine_steps']

                # Compute mean across steps
                cosine_array = np.array(cosine_steps)  # (num_steps, num_samples)
                mean_cosine_per_sample = cosine_array.mean(axis=0).tolist()

                batch_results[method_name][perturbation_method][layer_name] = mean_cosine_per_sample

        print(f"Completed cosine similarity evaluation for {method_name}")

    return batch_results


def internal_eval_attributions(args):
    """
    Main function to load saved attributions and evaluate intermediate ReLU layer attributions.

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

    # Load saved attributions
    print("Loading saved attributions...")
    saved_attributions = load_saved_attributions(args.attr_dir, dataset=args.dataset, model_name=args.model)

    if not saved_attributions:
        print("No saved attributions found!")
        return

    # Create wrapper for intermediate layer access
    wrapped_model = Wrapper(model, defense_mode="none")

    # Get sample input to initialize wrapper
    if args.dataset == "imagenet":
        sampled_indices = np.loadtxt("data_index_imagenet.txt", int, delimiter=",").tolist()
        sampled_subset = torch.utils.data.Subset(test, sampled_indices)
    else:
        sampled_indices = np.loadtxt("data_index_cub.txt", int, delimiter=",").tolist()
        sampled_subset = torch.utils.data.Subset(test, sampled_indices)

    test_loader = torch.utils.data.DataLoader(sampled_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    sample_input = next(iter(test_loader))[0][0].unsqueeze(0).to(args.device)
    wrapped_model.reorder_layers(sample_input)

    # Get intermediate ReLU layers
    print("Finding intermediate ReLU layers...")
    intermediate_layers = get_intermediate_relu_layers(model, args.model)

    # Process each batch
    all_results = {}
    all_overlap_results = {}
    all_correlation_results = {}
    all_cosine_results = {}

    for idx, (cur_input, _) in enumerate(test_loader):
        print(f"Processing batch {idx+1}/{len(test_loader)}")
        cur_input = cur_input.to(args.device)

        start_idx = args.batch_size * idx
        end_idx = start_idx + cur_input.shape[0]

        # Conditionally evaluate metrics based on args.metrics
        batch_results = None
        batch_overlap_results = None
        batch_correlation_results = None
        batch_cosine_results = None

        if 'mse' in args.metrics:
            # Evaluate intermediate activations using MSE loss
            print("Evaluating intermediate activations with MSE loss...")
            batch_results = evaluate_intermediate_activations_mse(
                model, cur_input, saved_attributions, intermediate_layers, start_idx, end_idx
            )

        if 'overlap' in args.metrics:
            # Evaluate unit overlap
            print("Evaluating unit overlap...")
            batch_overlap_results = evaluate_unit_overlap(
                model, cur_input, saved_attributions, intermediate_layers, start_idx, end_idx
            )

        if 'correlation' in args.metrics:
            # Evaluate correlation
            print("Evaluating correlation...")
            batch_correlation_results = evaluate_correlation(
                model, cur_input, saved_attributions, intermediate_layers, start_idx, end_idx
            )

        if 'cosine' in args.metrics:
            # Evaluate cosine similarity
            print("Evaluating cosine similarity...")
            batch_cosine_results = evaluate_cosine_similarity(
                model, cur_input, saved_attributions, intermediate_layers, start_idx, end_idx
            )

        # Accumulate MSE results
        if batch_results is not None:
            for method_name, method_results in batch_results.items():
                if method_name not in all_results:
                    all_results[method_name] = {}

                for perturbation_method, layer_results in method_results.items():
                    if perturbation_method not in all_results[method_name]:
                        all_results[method_name][perturbation_method] = {}

                    for layer_name, avg_mse_values in layer_results.items():
                        if layer_name not in all_results[method_name][perturbation_method]:
                            all_results[method_name][perturbation_method][layer_name] = []

                        all_results[method_name][perturbation_method][layer_name].extend(avg_mse_values)

        # Accumulate overlap results
        if batch_overlap_results is not None:
            for method_name, method_results in batch_overlap_results.items():
                if method_name not in all_overlap_results:
                    all_overlap_results[method_name] = {}

                for perturbation_method, layer_results in method_results.items():
                    if perturbation_method not in all_overlap_results[method_name]:
                        all_overlap_results[method_name][perturbation_method] = {}

                    for layer_name, overlap_dict in layer_results.items():
                        if layer_name not in all_overlap_results[method_name][perturbation_method]:
                            all_overlap_results[method_name][perturbation_method][layer_name] = {
                                'activated_stay_active_auc': [],
                                'inactivated_stay_inactive_auc': []
                            }

                        all_overlap_results[method_name][perturbation_method][layer_name]['activated_stay_active_auc'].extend(
                            overlap_dict['activated_stay_active_auc']
                        )
                        all_overlap_results[method_name][perturbation_method][layer_name]['inactivated_stay_inactive_auc'].extend(
                            overlap_dict['inactivated_stay_inactive_auc']
                        )

        # Accumulate correlation results
        if batch_correlation_results is not None:
            for method_name, method_results in batch_correlation_results.items():
                if method_name not in all_correlation_results:
                    all_correlation_results[method_name] = {}

                for perturbation_method, layer_results in method_results.items():
                    if perturbation_method not in all_correlation_results[method_name]:
                        all_correlation_results[method_name][perturbation_method] = {}

                    for layer_name, corr_auc_values in layer_results.items():
                        if layer_name not in all_correlation_results[method_name][perturbation_method]:
                            all_correlation_results[method_name][perturbation_method][layer_name] = []

                        all_correlation_results[method_name][perturbation_method][layer_name].extend(corr_auc_values)

        # Accumulate cosine similarity results
        if batch_cosine_results is not None:
            for method_name, method_results in batch_cosine_results.items():
                if method_name not in all_cosine_results:
                    all_cosine_results[method_name] = {}

                for perturbation_method, layer_results in method_results.items():
                    if perturbation_method not in all_cosine_results[method_name]:
                        all_cosine_results[method_name][perturbation_method] = {}

                    for layer_name, cosine_mean_values in layer_results.items():
                        if layer_name not in all_cosine_results[method_name][perturbation_method]:
                            all_cosine_results[method_name][perturbation_method][layer_name] = []

                        all_cosine_results[method_name][perturbation_method][layer_name].extend(cosine_mean_values)

    # Save results
    print("Saving results...")

    # Only save results for selected metrics
    if 'mse' in args.metrics and all_results:
        results_file = os.path.join(args.save_dir, "intermediate_eval_results.pth")
        torch.save(all_results, results_file)

        # Save CSV summary for MSE
        csv_path = os.path.join(args.save_dir, "intermediate_activation_mse_summary.csv")
        with open(csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["attribution_method", "perturbation_method", "layer_name", "avg_mse", "std_mse", "num_samples"]
            writer.writerow(header)

            for method_name, method_results in all_results.items():
                for perturbation_method, layer_results in method_results.items():
                    for layer_name, mse_values in layer_results.items():
                        if mse_values:
                            avg_mse = np.mean(mse_values)
                            std_mse = np.std(mse_values)
                            num_samples = len(mse_values)
                            writer.writerow([method_name, perturbation_method, layer_name, avg_mse, std_mse, num_samples])

        print(f"MSE results saved to {results_file}")
        print(f"MSE summary saved to {csv_path}")

    if 'overlap' in args.metrics and all_overlap_results:
        overlap_results_file = os.path.join(args.save_dir, "intermediate_overlap_results.pth")
        torch.save(all_overlap_results, overlap_results_file)

        # Save CSV summary for unit overlap (now with AUC)
        overlap_csv_path = os.path.join(args.save_dir, "intermediate_unit_overlap_summary.csv")
        with open(overlap_csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["attribution_method", "perturbation_method", "layer_name",
                      "avg_activated_auc", "std_activated_auc",
                      "avg_inactivated_auc", "std_inactivated_auc", "num_samples"]
            writer.writerow(header)

            for method_name, method_results in all_overlap_results.items():
                for perturbation_method, layer_results in method_results.items():
                    for layer_name, overlap_dict in layer_results.items():
                        activated_aucs = overlap_dict['activated_stay_active_auc']
                        inactivated_aucs = overlap_dict['inactivated_stay_inactive_auc']

                        if activated_aucs and inactivated_aucs:
                            avg_activated = np.mean(activated_aucs)
                            std_activated = np.std(activated_aucs)
                            avg_inactivated = np.mean(inactivated_aucs)
                            std_inactivated = np.std(inactivated_aucs)
                            num_samples = len(activated_aucs)
                            writer.writerow([method_name, perturbation_method, layer_name,
                                           avg_activated, std_activated, avg_inactivated, std_inactivated, num_samples])

        print(f"Overlap results saved to {overlap_results_file}")
        print(f"Overlap summary saved to {overlap_csv_path}")

    if 'correlation' in args.metrics and all_correlation_results:
        correlation_results_file = os.path.join(args.save_dir, "intermediate_correlation_results.pth")
        torch.save(all_correlation_results, correlation_results_file)

        # Save CSV summary for correlation (AUC)
        correlation_csv_path = os.path.join(args.save_dir, "intermediate_correlation_summary.csv")
        with open(correlation_csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["attribution_method", "perturbation_method", "layer_name",
                      "avg_correlation_auc", "std_correlation_auc", "num_samples"]
            writer.writerow(header)

            for method_name, method_results in all_correlation_results.items():
                for perturbation_method, layer_results in method_results.items():
                    for layer_name, corr_aucs in layer_results.items():
                        if corr_aucs:
                            avg_corr = np.mean(corr_aucs)
                            std_corr = np.std(corr_aucs)
                            num_samples = len(corr_aucs)
                            writer.writerow([method_name, perturbation_method, layer_name,
                                           avg_corr, std_corr, num_samples])

        print(f"Correlation results saved to {correlation_results_file}")
        print(f"Correlation summary saved to {correlation_csv_path}")

    if 'cosine' in args.metrics and all_cosine_results:
        cosine_results_file = os.path.join(args.save_dir, "intermediate_cosine_results.pth")
        torch.save(all_cosine_results, cosine_results_file)

        # Save CSV summary for cosine similarity (mean)
        cosine_csv_path = os.path.join(args.save_dir, "intermediate_cosine_summary.csv")
        with open(cosine_csv_path, mode='w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ["attribution_method", "perturbation_method", "layer_name",
                      "avg_cosine", "std_cosine", "num_samples"]
            writer.writerow(header)

            for method_name, method_results in all_cosine_results.items():
                for perturbation_method, layer_results in method_results.items():
                    for layer_name, cosine_means in layer_results.items():
                        if cosine_means:
                            avg_cosine = np.mean(cosine_means)
                            std_cosine = np.std(cosine_means)
                            num_samples = len(cosine_means)
                            writer.writerow([method_name, perturbation_method, layer_name,
                                           avg_cosine, std_cosine, num_samples])

        print(f"Cosine similarity results saved to {cosine_results_file}")
        print(f"Cosine similarity summary saved to {cosine_csv_path}")


def main():
    parser = argparse.ArgumentParser("internal_eval")
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--model', type=str, default='vgg16',
                       choices=["vgg19", "resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "googlenet", "alexnet"],
                       help='model architecture')
    parser.add_argument('--dataset', type=str, default='imagenet',
                       choices=["imagenet", "cub"], help='dataset to work on')
    parser.add_argument('--save_dir', type=str, default='./result_internal_eval', help='directory with saved attributions')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--attr_dir', type=str, default='./result_save', help='directory with saved attributions')
    parser.add_argument('--metrics', type=str, nargs='+',
                       default=['mse', 'overlap', 'correlation', 'cosine'],
                       choices=['mse', 'overlap', 'correlation', 'cosine'],
                       help='metrics to evaluate (can specify multiple: --metrics mse overlap correlation cosine)')

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.save_dir = f"{args.save_dir}/{args.dataset}/{args.model}"
    args.attr_dir = f"{args.attr_dir}/{args.dataset}/{args.model}"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print(f"Starting internal evaluation for {args.model} on {args.dataset}")
    print(f"Save directory: {args.save_dir}")

    internal_eval_attributions(args)


if __name__ == '__main__':
    main()
