from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch


@dataclass
class MaskHookHandle:
    module_name: str
    hook_handle: torch.utils.hooks.RemovableHandle

    def remove(self) -> None:
        self.hook_handle.remove()


class SVDGradMask:
    """Mask gradients in singular-direction space for a projection weight."""

    def __init__(
        self,
        u: torch.Tensor,
        vh: torch.Tensor,
        bouncer_sv_indices: Sequence[int],
    ):
        self.u = u.detach()
        self.vh = vh.detach()
        self.bouncer_sv_indices = torch.tensor(list(sorted(set(int(i) for i in bouncer_sv_indices))), dtype=torch.long)

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        if self.bouncer_sv_indices.numel() == 0:
            return grad

        u = self.u.to(device=grad.device, dtype=grad.dtype)
        vh = self.vh.to(device=grad.device, dtype=grad.dtype)
        idx = self.bouncer_sv_indices.to(device=grad.device)

        grad_svd = u.transpose(0, 1) @ grad @ vh.transpose(0, 1)
        grad_svd[idx, :] = 0.0
        grad_svd[:, idx] = 0.0
        return u @ grad_svd @ vh


class DimensionGradMask:
    """Fallback mask that zeros gradient columns for selected input dimensions."""

    def __init__(self, bouncer_col_indices: Sequence[int]):
        self.bouncer_col_indices = torch.tensor(list(sorted(set(int(i) for i in bouncer_col_indices))), dtype=torch.long)

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        if self.bouncer_col_indices.numel() == 0:
            return grad
        masked = grad.clone()
        idx = self.bouncer_col_indices.to(device=grad.device)
        if masked.ndim == 2:
            masked[:, idx] = 0.0
        else:
            masked[..., idx] = 0.0
        return masked


def install_weight_grad_hook(module: torch.nn.Module, hook_fn) -> torch.utils.hooks.RemovableHandle:
    if not hasattr(module, "weight"):
        raise AttributeError(f"Module {module.__class__.__name__} has no weight")
    return module.weight.register_hook(hook_fn)
