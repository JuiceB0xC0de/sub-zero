from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .atlas import BrainAtlas
from .hooks import DimensionGradMask, SVDGradMask, install_weight_grad_hook
from .model_utils import get_projection_map, resolve_layers


@dataclass
class SubZeroHandle:
    original_weights: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)
    hook_handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def remove(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def restore(self, model: torch.nn.Module) -> None:
        layers = resolve_layers(model)
        for (layer_idx, pname), w in self.original_weights.items():
            pmap = get_projection_map(layers[layer_idx])
            mod = pmap.get(pname)
            if mod is None:
                continue
            with torch.no_grad():
                mod.weight.data.copy_(w.to(device=mod.weight.device, dtype=mod.weight.dtype))


def _verify_svd_roundtrip(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    recon = u @ torch.diag(s) @ vh
    drift = float(torch.max(torch.abs(recon - w)).item())
    return u, s, vh, drift


def apply_sub_zero(
    model: torch.nn.Module,
    atlas: BrainAtlas,
    sacred_layers: Optional[Sequence[int]] = None,
    svd_drift_threshold: float = 1e-4,
) -> SubZeroHandle:
    layers = resolve_layers(model)
    selected = set(int(x) for x in (sacred_layers if sacred_layers is not None else atlas.sacred_layers))
    handle = SubZeroHandle()

    for li in sorted(selected):
        layer_atlas = atlas.layers.get(li)
        if layer_atlas is None:
            continue

        pmap = get_projection_map(layers[li])
        for pname, p_atlas in layer_atlas.per_projection.items():
            mod = pmap.get(pname)
            if mod is None or not hasattr(mod, "weight"):
                continue
            if p_atlas.bouncer_sv_indices.numel() == 0:
                continue

            w = mod.weight.data.detach().float().cpu()
            u, s, vh, drift = _verify_svd_roundtrip(w)
            scales = p_atlas.per_direction_target_scale.detach().float().cpu()
            if scales.numel() != s.numel():
                continue

            handle.original_weights[(li, pname)] = mod.weight.data.detach().clone()

            s_new = s * scales
            w_new = u @ torch.diag(s_new) @ vh
            with torch.no_grad():
                mod.weight.data.copy_(w_new.to(device=mod.weight.device, dtype=mod.weight.dtype))

            if drift <= svd_drift_threshold:
                hook_fn = SVDGradMask(u, vh, p_atlas.bouncer_sv_indices.tolist())
            else:
                # fallback: map bouncer singular vectors to top input-dim columns by magnitude
                idx_cols: List[int] = []
                for sv_idx in p_atlas.bouncer_sv_indices.tolist():
                    vec = vh[int(sv_idx)]
                    idx_cols.extend(torch.topk(torch.abs(vec), k=min(8, vec.numel())).indices.tolist())
                hook_fn = DimensionGradMask(idx_cols)

            h = install_weight_grad_hook(mod, hook_fn)
            handle.hook_handles.append(h)

    return handle
