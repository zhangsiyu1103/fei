"""Clean-support channel intervention and pathway-independence ratio."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from torch import nn


@dataclass(frozen=True)
class ChannelSplit:
    """A model decomposed at a channel-bearing hidden representation."""

    model: nn.Module
    prefix_fn: Callable[[torch.Tensor], torch.Tensor]
    suffix_fn: Callable[[torch.Tensor], torch.Tensor]

    def prefix(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.prefix_fn(inputs)

    def suffix(self, activations: torch.Tensor) -> torch.Tensor:
        return self.suffix_fn(activations)


@dataclass(frozen=True)
class PathwayResult:
    """Per-image output of the clean-support intervention."""

    rho: float
    intact_auc: float
    ablated_auc: float
    selected_channels: Tuple[int, ...]
    clean_probability: float
    selected_probability: float
    reached_threshold: bool
    split_max_abs_error: float


def target_probability(logits: torch.Tensor, target: int) -> torch.Tensor:
    targets = torch.full(
        (len(logits),), target, dtype=torch.long, device=logits.device
    )
    return torch.softmax(logits, dim=1).gather(1, targets[:, None]).squeeze(1)


@torch.inference_mode()
def select_clean_channels(
    split: ChannelSplit,
    image: torch.Tensor,
    target: int,
    *,
    threshold: float = 0.5,
    candidate_batch_size: int = 64,
    max_selected: Optional[int] = None,
) -> Tuple[float, Tuple[int, ...], Tuple[float, ...], float]:
    """Greedily select channels until target probability falls below threshold.

    Selection uses only the clean image. At each step, the routine chooses the
    remaining channel whose additional zero ablation minimizes target-class
    softmax probability.
    """
    if image.ndim != 4 or len(image) != 1:
        raise ValueError("select_clean_channels expects one image shaped [1,C,H,W]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive")

    clean_logits = split.model(image)
    clean_probability = float(target_probability(clean_logits, target).item())
    activation = split.prefix(image)
    if activation.ndim < 2:
        raise ValueError("the split activation must have a channel dimension")
    split_logits = split.suffix(activation)
    split_error = float((clean_logits - split_logits).abs().max().item())

    current = activation.clone()
    channel_count = current.shape[1]
    remaining = torch.ones(channel_count, dtype=torch.bool, device=image.device)
    selected: List[int] = []
    probabilities: List[float] = []
    limit = channel_count if max_selected is None else min(channel_count, max_selected)

    while len(selected) < limit:
        candidates = torch.nonzero(remaining, as_tuple=False).flatten()
        if candidates.numel() == 0:
            break
        best_probability = math.inf
        best_channel = -1
        for start in range(0, candidates.numel(), candidate_batch_size):
            chunk = candidates[start : start + candidate_batch_size]
            candidate_activations = current.expand(
                chunk.numel(), *current.shape[1:]
            ).clone()
            candidate_activations[
                torch.arange(chunk.numel(), device=image.device), chunk
            ] = 0
            probabilities_chunk = target_probability(
                split.suffix(candidate_activations), target
            )
            offset = int(torch.argmin(probabilities_chunk).item())
            probability = float(probabilities_chunk[offset].item())
            if probability < best_probability:
                best_probability = probability
                best_channel = int(chunk[offset].item())

        if best_channel < 0:
            raise RuntimeError("no channel candidate was selected")
        current[:, best_channel] = 0
        remaining[best_channel] = False
        selected.append(best_channel)
        probabilities.append(best_probability)
        if best_probability <= threshold * clean_probability:
            break

    return (
        clean_probability,
        tuple(selected),
        tuple(probabilities),
        split_error,
    )


def insertion_trajectory(
    image: torch.Tensor,
    attribution: torch.Tensor,
    step: int,
    substrate: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Construct the discrete insertion sweep used by the diagnostic."""
    if image.ndim != 4 or len(image) != 1:
        raise ValueError("insertion_trajectory expects one image shaped [1,C,H,W]")
    if step < 1:
        raise ValueError("step must be positive")
    map_2d = attribution.squeeze()
    if map_2d.ndim == 3:
        map_2d = map_2d.mean(dim=0)
    if map_2d.ndim != 2 or map_2d.shape != image.shape[-2:]:
        raise ValueError("attribution must reduce to the input's [H,W] shape")

    _, channels, height, width = image.shape
    pixels = height * width
    order = torch.argsort(map_2d.flatten(), descending=True)
    n_steps = math.ceil(pixels / step)
    frames = torch.empty(
        (n_steps + 1, channels, height, width),
        dtype=image.dtype,
        device=image.device,
    )
    if substrate is None:
        frames[0].zero_()
    else:
        if substrate.shape != image.shape:
            raise ValueError("substrate must have the same shape as image")
        frames[0].copy_(substrate[0])

    flat_frames = frames.view(n_steps + 1, channels, pixels)
    flat_image = image[0].view(channels, pixels)
    for index in range(1, n_steps + 1):
        flat_frames[index].copy_(flat_frames[index - 1])
        coordinates = order[(index - 1) * step : min(index * step, pixels)]
        flat_frames[index, :, coordinates] = flat_image[:, coordinates]
    return frames


@torch.inference_mode()
def _paired_curves(
    split: ChannelSplit,
    frames: torch.Tensor,
    target: int,
    channels: Sequence[int],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    intact_values = []
    ablated_values = []
    for start in range(0, len(frames), batch_size):
        activation = split.prefix(frames[start : start + batch_size])
        intact_values.append(target_probability(split.suffix(activation), target).cpu())
        ablated = activation.clone()
        ablated[:, list(channels)] = 0
        ablated_values.append(target_probability(split.suffix(ablated), target).cpu())
    return torch.cat(intact_values), torch.cat(ablated_values)


def _normalized_auc(values: torch.Tensor) -> float:
    if len(values) < 2:
        return float(values[0].item())
    x_axis = torch.linspace(0.0, 1.0, len(values), dtype=values.dtype)
    return float(torch.trapezoid(values, x=x_axis).item())


@torch.inference_mode()
def pathway_ratio(
    split: ChannelSplit,
    image: torch.Tensor,
    attribution: torch.Tensor,
    target: int,
    *,
    threshold: float = 0.5,
    insertion_step: int = 1792,
    candidate_batch_size: int = 64,
    forward_batch_size: int = 64,
    substrate: Optional[torch.Tensor] = None,
    auc_floor: float = 1e-8,
) -> PathwayResult:
    """Compute intact insertion AUC, ablated AUC, and ``rho=A_-S/A``."""
    clean_probability, channels, probabilities, split_error = select_clean_channels(
        split,
        image,
        target,
        threshold=threshold,
        candidate_batch_size=candidate_batch_size,
    )
    frames = insertion_trajectory(image, attribution, insertion_step, substrate)
    intact_curve, ablated_curve = _paired_curves(
        split, frames, target, channels, forward_batch_size
    )
    intact_auc = _normalized_auc(intact_curve)
    ablated_auc = _normalized_auc(ablated_curve)
    rho = ablated_auc / intact_auc if intact_auc > auc_floor else float("nan")
    selected_probability = probabilities[-1] if probabilities else clean_probability
    return PathwayResult(
        rho=rho,
        intact_auc=intact_auc,
        ablated_auc=ablated_auc,
        selected_channels=channels,
        clean_probability=clean_probability,
        selected_probability=selected_probability,
        reached_threshold=selected_probability <= threshold * clean_probability,
        split_max_abs_error=split_error,
    )


@dataclass(frozen=True)
class TokenSplit:
    """A transformer model decomposed at a token-sequence hidden representation.

    ``prefix_fn`` maps an input image batch to the token sequence
    ``[B, N, D]`` (including any leading tokens such as CLS); ``suffix_fn``
    completes the forward pass from that sequence to logits. ``cls_offset``
    is the number of leading tokens excluded from ablation (1 for a single
    CLS token, the standard ViT/DeiT layout).
    """

    model: nn.Module
    prefix_fn: Callable[[torch.Tensor], torch.Tensor]
    suffix_fn: Callable[[torch.Tensor], torch.Tensor]
    cls_offset: int = 1

    def prefix(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.prefix_fn(inputs)

    def suffix(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.suffix_fn(tokens)


@dataclass(frozen=True)
class TokenPathwayResult:
    """Per-image output of the DeiT token-level clean-support intervention."""

    rho_token: float
    intact_auc: float
    ablated_auc: float
    selected_tokens: Tuple[int, ...]
    clean_probability: float
    selected_probability: float
    reached_threshold: bool
    split_max_abs_error: float


@torch.inference_mode()
def select_clean_tokens(
    split: TokenSplit,
    image: torch.Tensor,
    target: int,
    *,
    threshold: float = 0.5,
    candidate_batch_size: int = 64,
    max_selected: Optional[int] = None,
) -> Tuple[float, Tuple[int, ...], Tuple[float, ...], float]:
    """Greedily select patch tokens until target probability falls below threshold.

    Mirrors ``select_clean_channels`` with the ablation unit changed from a
    convolutional channel to a patch-token embedding (zeroed in full), the
    DeiT analogue of clean-channel selection: the same clean top-1 target,
    the same greedy steepest-descent rule, and the same stopping threshold.
    """
    if image.ndim != 4 or len(image) != 1:
        raise ValueError("select_clean_tokens expects one image shaped [1,C,H,W]")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must lie in [0, 1]")
    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be positive")

    clean_logits = split.model(image)
    clean_probability = float(target_probability(clean_logits, target).item())
    activation = split.prefix(image)
    if activation.ndim != 3:
        raise ValueError("the token split activation must have shape [1, N, D]")
    split_logits = split.suffix(activation)
    split_error = float((clean_logits - split_logits).abs().max().item())

    current = activation.clone()
    token_count = current.shape[1] - split.cls_offset
    if token_count < 1:
        raise ValueError("no patch tokens remain after excluding cls_offset")
    remaining = torch.ones(token_count, dtype=torch.bool, device=image.device)
    selected: List[int] = []
    probabilities: List[float] = []
    limit = token_count if max_selected is None else min(token_count, max_selected)

    while len(selected) < limit:
        candidates = torch.nonzero(remaining, as_tuple=False).flatten()
        if candidates.numel() == 0:
            break
        best_probability = math.inf
        best_token = -1
        for start in range(0, candidates.numel(), candidate_batch_size):
            chunk = candidates[start : start + candidate_batch_size]
            candidate_activations = current.expand(
                chunk.numel(), *current.shape[1:]
            ).clone()
            token_indices = chunk + split.cls_offset
            candidate_activations[
                torch.arange(chunk.numel(), device=image.device), token_indices, :
            ] = 0
            probabilities_chunk = target_probability(
                split.suffix(candidate_activations), target
            )
            offset = int(torch.argmin(probabilities_chunk).item())
            probability = float(probabilities_chunk[offset].item())
            if probability < best_probability:
                best_probability = probability
                best_token = int(chunk[offset].item())

        if best_token < 0:
            raise RuntimeError("no token candidate was selected")
        current[:, best_token + split.cls_offset, :] = 0
        remaining[best_token] = False
        selected.append(best_token)
        probabilities.append(best_probability)
        if best_probability <= threshold * clean_probability:
            break

    return (
        clean_probability,
        tuple(selected),
        tuple(probabilities),
        split_error,
    )


@torch.inference_mode()
def _paired_curves_tokens(
    split: TokenSplit,
    frames: torch.Tensor,
    target: int,
    tokens: Sequence[int],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    intact_values = []
    ablated_values = []
    token_indices = [split.cls_offset + t for t in tokens]
    for start in range(0, len(frames), batch_size):
        activation = split.prefix(frames[start : start + batch_size])
        intact_values.append(target_probability(split.suffix(activation), target).cpu())
        ablated = activation.clone()
        ablated[:, token_indices, :] = 0
        ablated_values.append(target_probability(split.suffix(ablated), target).cpu())
    return torch.cat(intact_values), torch.cat(ablated_values)


@torch.inference_mode()
def token_pathway_ratio(
    split: TokenSplit,
    image: torch.Tensor,
    attribution: torch.Tensor,
    target: int,
    *,
    threshold: float = 0.5,
    insertion_step: int = 1792,
    candidate_batch_size: int = 64,
    forward_batch_size: int = 64,
    substrate: Optional[torch.Tensor] = None,
    auc_floor: float = 1e-8,
) -> TokenPathwayResult:
    """Compute intact insertion AUC, ablated AUC, and ``rho_token=A_-S/A``.

    The insertion trajectory is built exactly as for the CNN diagnostic
    (``insertion_trajectory``, operating on pixels via the attribution map);
    only the ablated unit changes, from a channel to a clean-supporting
    patch-token set selected by ``select_clean_tokens``.
    """
    clean_probability, tokens, probabilities, split_error = select_clean_tokens(
        split,
        image,
        target,
        threshold=threshold,
        candidate_batch_size=candidate_batch_size,
    )
    frames = insertion_trajectory(image, attribution, insertion_step, substrate)
    intact_curve, ablated_curve = _paired_curves_tokens(
        split, frames, target, tokens, forward_batch_size
    )
    intact_auc = _normalized_auc(intact_curve)
    ablated_auc = _normalized_auc(ablated_curve)
    rho_token = ablated_auc / intact_auc if intact_auc > auc_floor else float("nan")
    selected_probability = probabilities[-1] if probabilities else clean_probability
    return TokenPathwayResult(
        rho_token=rho_token,
        intact_auc=intact_auc,
        ablated_auc=ablated_auc,
        selected_tokens=tokens,
        clean_probability=clean_probability,
        selected_probability=selected_probability,
        reached_threshold=selected_probability <= threshold * clean_probability,
        split_max_abs_error=split_error,
    )


def timm_deit_token_split(model: nn.Module) -> TokenSplit:
    """Build the token split for a timm-style DeiT-Small model.

    Splits immediately after position-embedding addition and before the
    transformer blocks -- the interface used for the paper's DeiT
    token-intervention protocol. Assumes the standard timm
    ``VisionTransformer`` layout (``patch_embed``, ``cls_token``,
    ``pos_embed``, ``pos_drop``, ``blocks``, ``norm``, ``head``); adapt
    ``prefix_fn``/``suffix_fn`` directly for a different implementation.
    """

    def prefix_fn(inputs: torch.Tensor) -> torch.Tensor:
        tokens = model.patch_embed(inputs)
        cls_token = model.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = tokens + model.pos_embed
        return model.pos_drop(tokens)

    def suffix_fn(tokens: torch.Tensor) -> torch.Tensor:
        value = model.blocks(tokens)
        value = model.norm(value)
        return model.head(value[:, 0])

    return TokenSplit(model=model, prefix_fn=prefix_fn, suffix_fn=suffix_fn, cls_offset=1)


def _parse_index(layer_name: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}(?:\[|\.)(\d+)\]?", layer_name)
    if match is None:
        raise ValueError(f"expected {prefix}[N] or {prefix}.N, got {layer_name}")
    return int(match.group(1))


def torchvision_channel_split(
    model: nn.Module,
    model_kind: str,
    layer_name: Optional[str] = None,
) -> ChannelSplit:
    """Build the channel split used for VGG16, ResNet50, or ConvNeXt-Tiny."""
    model_kind = model_kind.lower().replace("_", "-")
    defaults = {
        "vgg16": "features[29]",
        "resnet50": "layer4[-1]",
        "convnext-tiny": "features[7]",
    }
    if model_kind not in defaults:
        raise ValueError("model_kind must be vgg16, resnet50, or convnext-tiny")
    layer_name = (layer_name or defaults[model_kind]).replace(" ", "")

    if model_kind in {"vgg16", "convnext-tiny"}:
        index = _parse_index(layer_name, "features")
        prefix = model.features[: index + 1]
        suffix = model.features[index + 1 :]
        if model_kind == "vgg16":
            def suffix_fn(activation: torch.Tensor) -> torch.Tensor:
                return model.classifier(
                    torch.flatten(model.avgpool(suffix(activation)), 1)
                )
        else:
            def suffix_fn(activation: torch.Tensor) -> torch.Tensor:
                return model.classifier(model.avgpool(suffix(activation)))
        return ChannelSplit(model, prefix, suffix_fn)

    match = re.fullmatch(r"layer([1-4])(?:\[-1\])?", layer_name)
    if match is None:
        raise ValueError("ResNet layer must be layer1[-1] through layer4[-1]")
    stage = int(match.group(1))

    def prefix_fn(inputs: torch.Tensor) -> torch.Tensor:
        value = model.maxpool(model.relu(model.bn1(model.conv1(inputs))))
        for index in range(1, stage + 1):
            value = getattr(model, f"layer{index}")(value)
        return value

    def suffix_fn(activation: torch.Tensor) -> torch.Tensor:
        value = activation
        for index in range(stage + 1, 5):
            value = getattr(model, f"layer{index}")(value)
        return model.fc(torch.flatten(model.avgpool(value), 1))

    return ChannelSplit(model, prefix_fn, suffix_fn)
