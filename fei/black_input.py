"""FGVis-derived all-black-input target-induction evaluation used in the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn

from .eqo import Target, generate_attribution, zero_reference


def imagenet_black_input_like(inputs: torch.Tensor) -> torch.Tensor:
    """Return an RGB-black canvas in ImageNet-normalized coordinates."""
    if inputs.ndim != 4 or inputs.shape[1] != 3:
        raise ValueError("inputs must have shape [B, 3, H, W]")
    mean = inputs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = inputs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((torch.zeros_like(inputs) - mean) / std).expand_as(inputs)


@dataclass(frozen=True)
class BlackInputInductionResult:
    """Outcome of the all-black-input evaluation for one target batch."""

    induced: torch.Tensor
    predicted: torch.Tensor
    perturbed: torch.Tensor
    attribution: torch.Tensor


def _target_tensor(target: Target, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(target, int):
        return torch.full((batch,), target, dtype=torch.long, device=device)
    tensor = torch.as_tensor(target, dtype=torch.long, device=device).reshape(-1)
    if tensor.numel() == 1 and batch > 1:
        tensor = tensor.expand(batch)
    if tensor.numel() != batch:
        raise ValueError("target must contain exactly one class index per input")
    return tensor


def evaluate_black_input_induction(
    model: nn.Module,
    black_inputs: torch.Tensor,
    target: Target,
    *,
    variant: str = "NONE",
    areas: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    iterations: int = 100,
    learning_rate: float = 0.05,
    area_weight: float = 0.1,
    model_kind: Optional[str] = None,
    deit_mode: str = "residual",
    use_amp: bool = False,
) -> BlackInputInductionResult:
    """Evaluate target induction from normalized all-black RGB inputs.

    This evaluation follows the black-input test introduced by FGVis (Wagner
    et al., CVPR 2019) and implements the paper's category-level protocol.
    ``black_inputs`` must
    be raw RGB-zero canvases after the model's standard normalization. EQO uses
    a zero tensor in normalized model-input space as its replacement reference;
    this is distinct from the normalized black RGB canvas. The returned
    ensemble attribution is applied to the canvas exactly as in the experiment,
    and induction occurs when the resulting top-1 prediction equals ``target``.

    For the aggregate ImageNet evaluation, repeat the same normalized black
    canvas and evaluate every category except the canvas's original top-1
    prediction.
    """
    if black_inputs.ndim != 4:
        raise ValueError("black_inputs must have shape [B, C, H, W]")
    target_tensor = _target_tensor(target, len(black_inputs), black_inputs.device)

    attribution = generate_attribution(
        model,
        black_inputs,
        target_tensor,
        variant=variant,
        areas=areas,
        iterations=iterations,
        learning_rate=learning_rate,
        area_weight=area_weight,
        reference_fn=zero_reference,
        model_kind=model_kind,
        deit_mode=deit_mode,
        use_amp=use_amp,
    )
    perturbed = black_inputs * attribution.expand_as(black_inputs)

    with torch.no_grad():
        predicted = model(perturbed).argmax(dim=1)
    induced = predicted == target_tensor
    return BlackInputInductionResult(
        induced=induced,
        predicted=predicted,
        perturbed=perturbed,
        attribution=attribution,
    )
