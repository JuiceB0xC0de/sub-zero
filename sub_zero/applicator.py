from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from .atlas import BrainAtlas
from .hooks import DASGradMask, DimensionGradMask, SVDGradMask, install_weight_grad_hook
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
    use_das: bool = True,
) -> SubZeroHandle:
    """Attenuate bouncer directions in projection weights.

    When `use_das=True` and the atlas has a DAS rotated basis for a (layer, proj),
    attenuate along the rotated subspace via:
        W_new = W - sum_r (1 - s_r) (W @ b_r) b_r^T
    and install a DASGradMask that projects out the bouncer subspace from gradient.
    Falls back to the SV-axis-aligned path when DAS isn't available.
    """
    layers = resolve_layers(model)
    selected = set(int(x) for x in (sacred_layers if sacred_layers is not None else atlas.sacred_layers))
    handle = SubZeroHandle()

    n_das = 0
    n_sv = 0
    n_skipped = 0

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
                n_skipped += 1
                continue

            w = mod.weight.data.detach().float().cpu()

            # ── DAS-aware branch: rotated subspace attenuation ────────────────
            if (
                use_das
                and p_atlas.bouncer_das_basis is not None
                and p_atlas.bouncer_das_target_scale is not None
            ):
                B = p_atlas.bouncer_das_basis.detach().float().cpu()       # [r, d_in]
                sc = p_atlas.bouncer_das_target_scale.detach().float().cpu()  # [r]
                if B.shape[1] == w.shape[1] and sc.numel() == B.shape[0]:
                    handle.original_weights[(li, pname)] = mod.weight.data.detach().clone()
                    # W_new = W - Σ_r (1 - s_r) (W b_r) b_r^T
                    attenuation = (1.0 - sc).unsqueeze(0)                  # [1, r]
                    Wb = w @ B.T                                            # [d_out, r]
                    delta = (Wb * attenuation) @ B                          # [d_out, d_in]
                    w_new = w - delta
                    with torch.no_grad():
                        mod.weight.data.copy_(
                            w_new.to(device=mod.weight.device, dtype=mod.weight.dtype)
                        )
                    h = install_weight_grad_hook(mod, DASGradMask(B))
                    handle.hook_handles.append(h)
                    n_das += 1
                    continue

            # ── SV-aligned fallback (legacy path) ─────────────────────────────
            u, s, vh, drift = _verify_svd_roundtrip(w)
            scales = p_atlas.per_direction_target_scale.detach().float().cpu()
            if scales.numel() != s.numel():
                n_skipped += 1
                continue

            handle.original_weights[(li, pname)] = mod.weight.data.detach().clone()

            s_new = s * scales
            w_new = u @ torch.diag(s_new) @ vh
            with torch.no_grad():
                mod.weight.data.copy_(w_new.to(device=mod.weight.device, dtype=mod.weight.dtype))

            if drift <= svd_drift_threshold:
                hook_fn = SVDGradMask(u, vh, p_atlas.bouncer_sv_indices.tolist())
            else:
                idx_cols: List[int] = []
                for sv_idx in p_atlas.bouncer_sv_indices.tolist():
                    vec = vh[int(sv_idx)]
                    idx_cols.extend(torch.topk(torch.abs(vec), k=min(8, vec.numel())).indices.tolist())
                hook_fn = DimensionGradMask(idx_cols)

            h = install_weight_grad_hook(mod, hook_fn)
            handle.hook_handles.append(h)
            n_sv += 1

    print(f"[sub-zero applicator] attenuated {n_das} via DAS, {n_sv} via SV, skipped {n_skipped}")
    return handle
