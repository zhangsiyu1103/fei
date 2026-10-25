"""Ensemble Quantile Optimization with optional ClipGrad constraints."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from .clipgrad import ClipGrad


ReferenceFn = Callable[[torch.Tensor], torch.Tensor]
Target = Union[int, Sequence[int], torch.Tensor]


def zero_reference(inputs: torch.Tensor) -> torch.Tensor:
    """Return a fixed zero reference in the input tensor's coordinate system."""
    return torch.zeros_like(inputs)


def random_solid_reference(inputs: torch.Tensor) -> torch.Tensor:
    """Sample an ImageNet-normalized solid RGB reference for each input."""
    batch, channels, height, width = inputs.shape
    if channels != 3:
        raise ValueError("random_solid_reference requires three-channel inputs")
    rgb = torch.randint(
        0,
        256,
        (batch, 3, 1, 1),
        dtype=inputs.dtype,
        device=inputs.device,
    ) / 255.0
    mean = inputs.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = inputs.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((rgb - mean) / std).expand(batch, 3, height, width)


def _target_tensor(target: Target, batch: int, device: torch.device) -> torch.Tensor:
    if isinstance(target, int):
        return torch.full((batch,), target, dtype=torch.long, device=device)
    return torch.as_tensor(target, dtype=torch.long, device=device).reshape(-1)


def _optimize_area(
    clipgrad: ClipGrad,
    inputs: torch.Tensor,
    target: torch.Tensor,
    area: float,
    start: torch.Tensor,
    iterations: int,
    learning_rate: float,
    area_weight: float,
    reference_fn: ReferenceFn,
    use_amp: bool,
) -> torch.Tensor:
    """Optimize one nonnegative mask increment above ``start``."""
    increment = torch.zeros_like(start, requires_grad=True)
    optimizer = torch.optim.Adam([increment], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, iterations)
    amp_enabled = use_amp and inputs.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if amp_enabled else None
    target_mass = int(start[0].numel() * area)
    max_increment = 1.0 - start

    clipgrad.enable_backward()
    try:
        for _ in range(iterations):
            mask = increment + start
            reference = reference_fn(inputs).to(
                device=inputs.device, dtype=inputs.dtype
            )
            if reference.shape != inputs.shape:
                raise ValueError("reference_fn must return a tensor shaped like inputs")

            area_loss = area_weight * (
                mask.sum(dim=(1, 2, 3)) - target_mass
            ).abs().mean()
            perturbed = inputs * mask + reference * (1.0 - mask)

            amp_context = torch.amp.autocast("cuda") if amp_enabled else nullcontext()
            with amp_context:
                output = clipgrad.model(perturbed)
                selected_score = output.float().gather(1, target[:, None]).mean()
            loss = area_loss - selected_score

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()

            with torch.no_grad():
                increment.clamp_(min=0)
                increment.copy_(torch.minimum(increment, max_increment))
    finally:
        clipgrad.remove_backward_hooks()

    return (increment + start).detach()


def generate_attribution(
    model: nn.Module,
    inputs: torch.Tensor,
    target: Target,
    *,
    variant: str = "VM",
    areas: Sequence[float] = (0.1, 0.3, 0.5, 0.7, 0.9),
    iterations: int = 100,
    learning_rate: float = 0.1,
    area_weight: float = 0.1,
    reference_fn: ReferenceFn = random_solid_reference,
    model_kind: Optional[str] = None,
    deit_mode: str = "attention",
    use_amp: bool = False,
    return_stages: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]]:
    """Generate an EQO/FEI attribution map.

    ``variant='NONE'`` runs Unconstrained EQO. The four ClipGrad choices are
    ``VM``, ``IBM``, ``IVM``, and ``AVM``. Area levels must be strictly
    increasing because each stage is initialized from the preceding mask.
    """
    if inputs.ndim != 4:
        raise ValueError("inputs must have shape [B, C, H, W]")
    if not areas:
        raise ValueError("areas must contain at least one retained-area level")
    if any(not 0.0 <= area <= 1.0 for area in areas):
        raise ValueError("every area must lie in [0, 1]")
    if any(left >= right for left, right in zip(areas, areas[1:])):
        raise ValueError("areas must be strictly increasing")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = inputs.device
    if inputs.device != model_device:
        raise ValueError(
            f"model is on {model_device}, but inputs are on {inputs.device}"
        )

    target_tensor = _target_tensor(target, len(inputs), inputs.device)
    if target_tensor.numel() != len(inputs):
        raise ValueError("target must contain exactly one class index per input")

    previous_training = model.training
    parameter_states = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    clipgrad = ClipGrad(
        model,
        variant=variant,
        model_kind=model_kind,
        deit_mode=deit_mode,
    )
    current = torch.zeros(
        (len(inputs), 1, inputs.shape[-2], inputs.shape[-1]),
        dtype=inputs.dtype,
        device=inputs.device,
    )
    stages = []
    try:
        clipgrad.capture_clean(inputs)
        for area in areas:
            current = _optimize_area(
                clipgrad,
                inputs,
                target_tensor,
                area,
                current,
                iterations,
                learning_rate,
                area_weight,
                reference_fn,
                use_amp,
            )
            stages.append(current)
        attribution = torch.stack(stages, dim=0).mean(dim=0)
    finally:
        clipgrad.close()
        for parameter, requires_grad in zip(model.parameters(), parameter_states):
            parameter.requires_grad_(requires_grad)
        model.train(previous_training)

    if return_stages:
        return attribution, tuple(stages)
    return attribution
