from __future__ import annotations

from typing import Dict, Optional

import torch

from .atlas import BrainAtlas
from .model_utils import resolve_layers


class SubZeroWandbLogger:
    def __init__(self, atlas: BrainAtlas, wandb_run=None):
        self.atlas = atlas
        self.wandb_run = wandb_run

    def build_static_payload(self) -> Dict[str, float]:
        payload: Dict[str, float] = {}
        for li, layer in self.atlas.layers.items():
            payload[f"sub_zero/classifier_accuracy/layer_{li}"] = float(layer.classifier_accuracy)
            payload[f"sub_zero/corp_refusal_angle_deg/layer_{li}"] = float(layer.angle_degrees)
            total = 0
            bouncers = 0
            for p in layer.per_projection.values():
                total += int(p.S.numel())
                bouncers += int(p.bouncer_sv_indices.numel())
            payload[f"sub_zero/bouncer_pct/layer_{li}"] = (bouncers / total) if total > 0 else 0.0
        return payload

    def log_static(self, step: int = 0) -> None:
        if self.wandb_run is None:
            return
        self.wandb_run.log(self.build_static_payload(), step=step)

    def log_step_alignment(
        self,
        model: torch.nn.Module,
        layer_activations: Dict[int, torch.Tensor],
        step: int,
    ) -> None:
        if self.wandb_run is None:
            return
        payload: Dict[str, float] = {}
        for li, acts in layer_activations.items():
            layer = self.atlas.layers.get(li)
            if layer is None or acts.numel() == 0:
                continue
            v = acts.mean(dim=0).detach().float().cpu()
            corp = layer.corporate_axis_clean.float().cpu()
            ref = layer.refusal_axis.float().cpu()
            payload[f"sub_zero/corp_axis_alignment/layer_{li}"] = float(torch.nn.functional.cosine_similarity(v, corp, dim=0).item())
            if ref.norm().item() > 0:
                payload[f"sub_zero/refusal_axis_alignment/layer_{li}"] = float(torch.nn.functional.cosine_similarity(v, ref, dim=0).item())
        if payload:
            self.wandb_run.log(payload, step=step)
