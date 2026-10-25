"""ClipGrad activation capture and backward-gradient constraints."""

from __future__ import annotations

from typing import List, Optional

import torch
from torch import nn


SUPPORTED_VARIANTS = {"NONE", "VM", "IBM", "IVM", "AVM"}


class ClipGrad:
    """Attach the ClipGrad rule to a frozen attribution model.

    A clean forward pass first records the selected activations. During mask
    optimization, forward hooks compare perturbed and clean activations, and
    backward hooks apply the selected coordinate-wise gradient rule.
    """

    def __init__(
        self,
        model: nn.Module,
        variant: str,
        model_kind: Optional[str] = None,
        deit_mode: str = "residual",
    ) -> None:
        variant = variant.upper()
        if variant not in SUPPORTED_VARIANTS:
            choices = ", ".join(sorted(SUPPORTED_VARIANTS))
            raise ValueError(f"Unknown ClipGrad variant {variant!r}; choose {choices}")

        self.model = model
        self.variant = variant
        self.model_kind = (model_kind or "").lower().replace("_", "-")
        self.deit_mode = deit_mode
        self.device = self._model_device()

        self.layers: List[nn.Module] = []
        self.blocks: List[nn.Module] = []
        self.attentions: List[nn.Module] = []
        self.gelus: List[nn.Module] = []
        self.forward_handles: List[torch.utils.hooks.RemovableHandle] = []
        self.backward_handles: List[torch.utils.hooks.RemovableHandle] = []

        for name, module in model.named_modules():
            children = list(module.children())
            if not children:
                self.layers.append(module)
                if isinstance(module, nn.GELU) and (
                    ".mlp.act" in name or self.is_convnext
                ):
                    self.gelus.append(module)
            else:
                type_name = type(module).__name__
                if "Block" in type_name and "blocks" in name:
                    self.blocks.append(module)
                if "Attention" in type_name and ".attn" in name:
                    self.attentions.append(module)

        self.clean_activations: List[torch.Tensor] = []
        self.clip_masks: List[torch.Tensor] = []
        self.activation_index = 0
        self.activation_count = 0
        self.start = 0
        self.end = len(self.layers)

    @property
    def is_deit(self) -> bool:
        return self.model_kind == "deit-small"

    @property
    def is_convnext(self) -> bool:
        return self.model_kind == "convnext-tiny"

    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _deit_modules(self) -> List[nn.Module]:
        if self.deit_mode == "attention":
            return self.attentions
        if self.deit_mode == "gelu":
            return self.gelus
        if self.deit_mode == "both":
            return self.attentions + self.blocks
        return self.blocks

    def _selected_tensor_modules(self) -> List[nn.Module]:
        if self.is_deit:
            return self._deit_modules()
        if self.is_convnext:
            return self.gelus
        return []

    def _selected_activation_layers(self) -> List[nn.Module]:
        selected: List[nn.Module] = []
        for layer in self.layers[self.start : self.end]:
            if isinstance(layer, nn.Linear) and not self.is_convnext:
                break
            if isinstance(layer, (nn.ReLU, nn.GELU)):
                if isinstance(layer, nn.ReLU):
                    layer.inplace = False
                selected.append(layer)
        return selected

    def _capture_clean(self, _module, _inputs, output) -> None:
        self.clean_activations.append(output.detach().clone())

    def capture_clean(
        self,
        inputs: torch.Tensor,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> None:
        """Record clean activations and install perturbed-forward hooks."""
        if self.variant == "NONE":
            return
        self.start = 0 if start is None else start
        self.end = len(self.layers) if end is None else end

        modules = (
            self._selected_tensor_modules()
            if (self.is_deit or self.is_convnext)
            else self._selected_activation_layers()
        )
        if not modules:
            raise ValueError(
                "ClipGrad found no supported activation sites. For ConvNeXt or "
                "DeiT, pass model_kind='convnext-tiny' or 'deit-small'."
            )

        self.clean_activations = []
        capture_handles = [m.register_forward_hook(self._capture_clean) for m in modules]
        try:
            with torch.no_grad():
                self.model(inputs.to(self.device))
        finally:
            for handle in capture_handles:
                handle.remove()

        self.activation_count = len(self.clean_activations)
        self.activation_index = 0
        self.clip_masks = []
        if self.is_deit or self.is_convnext:
            self.forward_handles = [
                module.register_forward_hook(self._tensor_forward_hook)
                for module in modules
            ]
        else:
            self.forward_handles = [
                module.register_forward_hook(self._module_forward_hook)
                for module in modules
            ]

    def _activation_mask(
        self, perturbed: torch.Tensor, clean: torch.Tensor
    ) -> torch.Tensor:
        if self.variant == "IVM":
            return perturbed <= clean
        if self.variant in {"AVM", "VM"}:
            return perturbed >= clean
        if self.variant == "IBM":
            return clean > 0
        raise RuntimeError(f"Unsupported ClipGrad variant: {self.variant}")

    def _next_mask(self, output: torch.Tensor) -> torch.Tensor:
        if self.activation_index == self.activation_count:
            self.activation_index = 0
        clean = self.clean_activations[self.activation_index]
        mask = self._activation_mask(output, clean).detach()
        if len(self.clip_masks) == self.activation_count:
            self.clip_masks[self.activation_index] = mask
        else:
            self.clip_masks.append(mask)
        self.activation_index += 1
        return mask

    def _module_forward_hook(self, _module, _inputs, output) -> None:
        self._next_mask(output)

    def _tensor_forward_hook(self, _module, _inputs, output) -> None:
        mask = self._next_mask(output)
        if output.requires_grad:
            self.backward_handles.append(
                output.register_hook(self._tensor_backward_hook(mask))
            )

    def _tensor_backward_hook(self, activation_mask: torch.Tensor):
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            return self._apply_rule(gradient, activation_mask.to(gradient.device))

        return hook

    def _apply_rule(
        self, gradient: torch.Tensor, activation_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.variant in {"IBM", "IVM"}:
            keep = (gradient > 0) | activation_mask
        elif self.variant == "AVM":
            keep = (gradient < 0) | activation_mask
        elif self.variant == "VM":
            positive = gradient > 0
            keep = (positive & activation_mask) | (~positive & ~activation_mask)
        else:
            raise RuntimeError(f"Unsupported ClipGrad variant: {self.variant}")
        return gradient * keep

    def _module_backward_hook(self, _module, grad_input, grad_output):
        if self.activation_index == 0:
            self.activation_index = self.activation_count
        self.activation_index -= 1
        mask = self.clip_masks[self.activation_index].to(self.device)

        if self.variant == "AVM":
            source = grad_output[0]
            keep = (source < 0) | mask
            return (grad_input[0] * keep,)
        return (self._apply_rule(grad_input[0], mask),)

    def enable_backward(self) -> None:
        """Install module backward hooks for the next optimization stage."""
        if self.variant == "NONE" or self.is_deit or self.is_convnext:
            return
        self.remove_backward_hooks()
        self.backward_handles = [
            layer.register_full_backward_hook(self._module_backward_hook)
            for layer in self._selected_activation_layers()
        ]

    def remove_backward_hooks(self) -> None:
        for handle in self.backward_handles:
            handle.remove()
        self.backward_handles = []

    def close(self) -> None:
        """Remove every hook owned by this object."""
        for handle in self.forward_handles:
            handle.remove()
        self.remove_backward_hooks()
        self.forward_handles = []
        self.clean_activations = []
        self.clip_masks = []
        self.activation_index = 0
        self.activation_count = 0

