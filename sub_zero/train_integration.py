from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import torch

from .applicator import SubZeroHandle, apply_sub_zero
from .atlas import BrainAtlas
from .probe import ProbeConfig, build_atlas


@dataclass
class SubZeroRuntime:
    atlas: BrainAtlas
    handle: SubZeroHandle

    def close(self, model: torch.nn.Module) -> None:
        self.handle.restore(model)
        self.handle.remove()


def setup_sub_zero(
    model: torch.nn.Module,
    tokenizer,
    probe_config: ProbeConfig,
    task_batches: Sequence[Dict[str, torch.Tensor]],
    cache_path: Optional[str] = None,
    sacred_layers: Optional[Sequence[int]] = None,
) -> SubZeroRuntime:
    """Build/load atlas and apply Sub-Zero to model."""
    atlas = build_atlas(
        model,
        tokenizer,
        config=probe_config,
        task_batches=task_batches,
        cache_path=cache_path,
    )
    handle = apply_sub_zero(model, atlas, sacred_layers=sacred_layers)
    return SubZeroRuntime(atlas=atlas, handle=handle)
