"""Activation-trajectory similarity: cosine and MSE against the clean trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ActivationSimilarity:
    """Per-batch clean-vs-perturbed activation match at one or more layers."""

    mse: Tuple[float, ...]
    cosine: Tuple[float, ...]
    layer_names: Tuple[str, ...]


def _flatten(activation: torch.Tensor) -> torch.Tensor:
    return activation.reshape(activation.shape[0], -1).float()


@torch.inference_mode()
def activation_trajectory_similarity(
    model: nn.Module,
    clean_inputs: torch.Tensor,
    perturbed_inputs: torch.Tensor,
    layers: Sequence[nn.Module],
    layer_names: Sequence[str] = (),
) -> ActivationSimilarity:
    """Compare clean and perturbed activations at the given layers.

    Cosine similarity captures directional alignment; MSE captures magnitude
    preservation, matching the layer-wise comparison of ``h`` and ``h_tilde``
    in the paper. ``layers`` are forward-hooked in the order given; pass the
    same module used to build a ``ChannelSplit`` or ``TokenSplit`` to compare
    at the same interface used by the pathway diagnostic.

    Both input tensors must share a batch size; per-sample cosine similarity
    and per-element MSE are averaged over the batch.
    """
    if len(clean_inputs) != len(perturbed_inputs):
        raise ValueError("clean_inputs and perturbed_inputs must have matching batch size")
    if not layers:
        raise ValueError("layers must contain at least one module")
    names = tuple(layer_names) if layer_names else tuple(
        f"layer{i}" for i in range(len(layers))
    )
    if len(names) != len(layers):
        raise ValueError("layer_names must match layers in length")

    captured: Dict[int, torch.Tensor] = {}

    def make_hook(index: int):
        def hook(_module, _inputs, output):
            captured[index] = output.detach()

        return hook

    handles = [layer.register_forward_hook(make_hook(i)) for i, layer in enumerate(layers)]
    try:
        captured.clear()
        model(clean_inputs)
        clean_activations = [captured[i].clone() for i in range(len(layers))]
        captured.clear()
        model(perturbed_inputs)
        perturbed_activations = [captured[i].clone() for i in range(len(layers))]
    finally:
        for handle in handles:
            handle.remove()

    mse_values = []
    cosine_values = []
    for clean, perturbed in zip(clean_activations, perturbed_activations):
        clean_flat = _flatten(clean)
        perturbed_flat = _flatten(perturbed)
        if clean_flat.shape != perturbed_flat.shape:
            raise ValueError("clean and perturbed activations must match in shape")
        mse_values.append(float(((clean_flat - perturbed_flat) ** 2).mean().item()))
        cosine_values.append(
            float(
                torch.nn.functional.cosine_similarity(clean_flat, perturbed_flat, dim=1)
                .mean()
                .item()
            )
        )

    return ActivationSimilarity(mse=tuple(mse_values), cosine=tuple(cosine_values), layer_names=names)


def deit_cls_attention_map(attention: torch.Tensor) -> torch.Tensor:
    """Extract the head-averaged CLS-to-patch attention row.

    ``attention`` is the post-softmax final-layer attention tensor with shape
    ``[B, heads, tokens, tokens]`` (the standard timm ``Attention`` module
    convention). Returns ``[B, patch_tokens]``: the CLS row averaged over
    heads, with the CLS-to-CLS entry excluded and no spatial reshaping,
    matching the paper's last-attention activation-matching measurement.
    Feed the clean and perturbed maps this returns into
    ``activation_trajectory_similarity``-style cosine/MSE comparison, or
    compare them directly with the same formulas used there.
    """
    if attention.ndim != 4:
        raise ValueError("attention must have shape [B, heads, tokens, tokens]")
    if attention.shape[-1] < 2:
        raise ValueError("attention must include at least one patch token")
    cls_row = attention[:, :, 0, 1:]
    return cls_row.mean(dim=1)
