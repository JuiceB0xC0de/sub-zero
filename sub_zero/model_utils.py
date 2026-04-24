from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import torch


def resolve_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Resolve transformer layers with scheduler fallback when available."""
    base = model
    if hasattr(base, "module") and isinstance(base.module, torch.nn.Module):
        base = base.module
    if hasattr(base, "get_base_model"):
        try:
            base_model = base.get_base_model()
            if isinstance(base_model, torch.nn.Module):
                base = base_model
        except Exception:
            pass

    try:
        from lucky_pick_scheduler import resolve_transformer_layers  # type: ignore

        return list(resolve_transformer_layers(base))
    except Exception:
        pass

    paths = [
        "model.layers",
        "model.model.layers",
        "model.decoder.layers",
        "model.language_model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "text_model.layers",
        "decoder.layers",
        "transformer.layers",
        "transformer.h",
        "gpt_neox.layers",
        "layers",
    ]

    def _get_path(obj: Any, path: str):
        cur = obj
        for part in path.split("."):
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
        return cur

    for path in paths:
        layers = _get_path(base, path)
        if isinstance(layers, (list, tuple, torch.nn.ModuleList)) and len(layers) > 0:
            return list(layers)

    def _block_score(module: torch.nn.Module) -> int:
        score = 0
        if hasattr(module, "self_attn") or hasattr(module, "attention") or hasattr(module, "attn"):
            score += 2
        if hasattr(module, "mlp") or hasattr(module, "feed_forward") or hasattr(module, "ffn"):
            score += 2
        name = module.__class__.__name__.lower()
        if "decoder" in name or "block" in name or "layer" in name:
            score += 1
        return score

    best_layers = None
    best_score = float("-inf")
    for name, module in base.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        samples = list(module[: min(4, len(module))])
        child_score = sum(_block_score(child) for child in samples) / max(1, len(samples))
        path_score = 0.0
        lname = name.lower()
        if "language_model" in lname or "text" in lname or "decoder" in lname:
            path_score += 3.0
        if lname.endswith("layers") or lname.endswith(".layers") or lname.endswith(".h") or lname.endswith("blocks"):
            path_score += 2.0
        if "vision" in lname or "image" in lname or "audio" in lname or "encoder" in lname:
            path_score -= 4.0
        total_score = child_score + path_score + (0.01 * len(module))
        if total_score > best_score:
            best_score = total_score
            best_layers = module

    if best_layers is not None and len(best_layers) > 0:
        return list(best_layers)

    raise AttributeError("Could not resolve transformer layers")


def get_projection_map(layer: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """Return common projection modules for attention + mlp in a layer."""
    out: dict[str, torch.nn.Module] = {}

    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)
    if attn is not None:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(attn, name, None)
            if isinstance(mod, torch.nn.Module) and hasattr(mod, "weight"):
                out[name] = mod

    mlp = getattr(layer, "mlp", None) or getattr(layer, "feed_forward", None) or getattr(layer, "ffn", None)
    if mlp is not None:
        for name in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(mlp, name, None)
            if isinstance(mod, torch.nn.Module) and hasattr(mod, "weight"):
                out[name] = mod

    return out


def model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
