from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from .model_utils import resolve_layers, to_device


def _set_requires_grad(model: torch.nn.Module, value: bool) -> None:
    for p in model.parameters():
        p.requires_grad = value


def _layer_modules_for_score(layer: torch.nn.Module) -> List[torch.nn.Module]:
    mods: List[torch.nn.Module] = []
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)
    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None) or getattr(layer, "ffn", None)
    if attn is not None and hasattr(attn, "o_proj") and hasattr(attn.o_proj, "weight"):
        mods.append(attn.o_proj)
    if mlp is not None and hasattr(mlp, "down_proj") and hasattr(mlp.down_proj, "weight"):
        mods.append(mlp.down_proj)
    return mods


def run_aletheia(
    model: torch.nn.Module,
    task_batches: Sequence[Dict[str, torch.Tensor]],
    num_probe_batches: int = 5,
    chunk_size: int = 8,
    top_k_percent: float = 0.50,
) -> Tuple[List[int], Dict[int, float]]:
    """Gradient-guided layer ranking.

    task_batches should already contain labels for loss computation.
    """
    if not task_batches:
        raise ValueError("task_batches must be non-empty")

    device = next(model.parameters()).device
    layers = resolve_layers(model)
    n_layers = len(layers)
    score: Dict[int, float] = {i: 0.0 for i in range(n_layers)}

    was_training = model.training
    model.train()

    _set_requires_grad(model, False)

    n_use = min(int(max(1, num_probe_batches)), len(task_batches))

    for start in range(0, n_layers, chunk_size):
        end = min(start + chunk_size, n_layers)
        for li in range(start, end):
            for p in layers[li].parameters():
                p.requires_grad = True

        for bi in range(n_use):
            batch = to_device(task_batches[bi], device)
            model.zero_grad(set_to_none=True)
            out = model(**batch)
            loss = getattr(out, "loss", None)
            if loss is None:
                raise RuntimeError("Model output has no loss; provide labels in task_batches")
            loss.backward()

            for li in range(start, end):
                v = 0.0
                for mod in _layer_modules_for_score(layers[li]):
                    g = mod.weight.grad
                    if g is None:
                        continue
                    v += float(g.detach().float().norm().item())
                score[li] += v

        for li in range(start, end):
            for p in layers[li].parameters():
                p.requires_grad = False

    # Restore default trainability; caller controls explicit freezing policy.
    _set_requires_grad(model, True)

    if was_training:
        model.train()
    else:
        model.eval()

    ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)
    n_select = max(1, int(round(n_layers * float(top_k_percent))))
    sacred = [idx for idx, _ in ranked[:n_select]]
    return sacred, score
