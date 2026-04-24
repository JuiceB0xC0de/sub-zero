from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import torch


def trace_origin_layers(
    layer_projections: Dict[int, torch.Tensor],
    candidate_layers: Iterable[int],
    corr_threshold: float = 0.9,
) -> Dict[int, int]:
    """Heuristic origin trace for direction signal across neighboring layers.

    layer_projections: layer_idx -> tensor[num_samples] projection score for same direction.
    Returns mapping layer_idx -> origin_layer_idx.
    """
    out: Dict[int, int] = {}

    for layer_idx in sorted(candidate_layers):
        current = layer_projections.get(layer_idx)
        prev = layer_projections.get(layer_idx - 1)
        if current is None or prev is None:
            out[layer_idx] = layer_idx
            continue

        c = current.detach().float().cpu().numpy().reshape(-1)
        p = prev.detach().float().cpu().numpy().reshape(-1)
        if c.size != p.size or c.size < 3:
            out[layer_idx] = layer_idx
            continue

        corr = np.corrcoef(c, p)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        out[layer_idx] = layer_idx - 1 if corr > corr_threshold else layer_idx

    return out
