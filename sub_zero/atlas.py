from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch


@dataclass
class ProjectionAtlas:
    proj_name: str
    S: torch.Tensor
    bouncer_sv_indices: torch.Tensor
    per_direction_classifier_score: torch.Tensor
    per_direction_wanda_score: torch.Tensor
    per_direction_dark_variance: torch.Tensor
    per_direction_target_scale: torch.Tensor
    origin_layer: Dict[int, int] = field(default_factory=dict)
    # DAS rotation gate — principal causal axes within the bouncer subspace.
    # Each row is a unit direction in the projection's INPUT space, recoverable
    # as a linear combo of vh[bouncer_sv_indices]. None when DAS pass is off
    # or the candidate set was too small to decompose.
    bouncer_das_basis: "torch.Tensor | None" = None
    bouncer_das_explained: "torch.Tensor | None" = None    # variance ratios S²/ΣS²
    bouncer_das_singular_values: "torch.Tensor | None" = None  # raw S
    bouncer_das_weights: "torch.Tensor | None" = None      # [r, k] mixing matrix
    bouncer_das_target_scale: "torch.Tensor | None" = None # [r] per-axis attenuation
    # Stage 8 — capability orthogonality profile (per-axis loss-shift on capability corpora)
    bouncer_das_capability_profile: "Dict[str, torch.Tensor] | None" = None  # name → [r] tensor of |Δloss|
    bouncer_das_capability_damage: "torch.Tensor | None" = None  # [r] max |Δloss| over corpora
    bouncer_das_capability_passed: "torch.Tensor | None" = None  # [r] bool — capability-orthogonal

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "proj_name": self.proj_name,
            "S": self.S.detach().cpu(),
            "bouncer_sv_indices": self.bouncer_sv_indices.detach().cpu(),
            "per_direction_classifier_score": self.per_direction_classifier_score.detach().cpu(),
            "per_direction_wanda_score": self.per_direction_wanda_score.detach().cpu(),
            "per_direction_dark_variance": self.per_direction_dark_variance.detach().cpu(),
            "per_direction_target_scale": self.per_direction_target_scale.detach().cpu(),
            "origin_layer": {int(k): int(v) for k, v in self.origin_layer.items()},
        }
        if self.bouncer_das_basis is not None:
            d["bouncer_das_basis"] = self.bouncer_das_basis.detach().cpu()
        if self.bouncer_das_explained is not None:
            d["bouncer_das_explained"] = self.bouncer_das_explained.detach().cpu()
        if self.bouncer_das_singular_values is not None:
            d["bouncer_das_singular_values"] = self.bouncer_das_singular_values.detach().cpu()
        if self.bouncer_das_weights is not None:
            d["bouncer_das_weights"] = self.bouncer_das_weights.detach().cpu()
        if self.bouncer_das_target_scale is not None:
            d["bouncer_das_target_scale"] = self.bouncer_das_target_scale.detach().cpu()
        if self.bouncer_das_capability_profile is not None:
            d["bouncer_das_capability_profile"] = {
                k: v.detach().cpu() for k, v in self.bouncer_das_capability_profile.items()
            }
        if self.bouncer_das_capability_damage is not None:
            d["bouncer_das_capability_damage"] = self.bouncer_das_capability_damage.detach().cpu()
        if self.bouncer_das_capability_passed is not None:
            d["bouncer_das_capability_passed"] = self.bouncer_das_capability_passed.detach().cpu()
        return d

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ProjectionAtlas":
        return cls(
            proj_name=str(payload["proj_name"]),
            S=payload["S"],
            bouncer_sv_indices=payload["bouncer_sv_indices"],
            per_direction_classifier_score=payload["per_direction_classifier_score"],
            per_direction_wanda_score=payload["per_direction_wanda_score"],
            per_direction_dark_variance=payload["per_direction_dark_variance"],
            per_direction_target_scale=payload["per_direction_target_scale"],
            origin_layer={int(k): int(v) for k, v in dict(payload.get("origin_layer", {})).items()},
            bouncer_das_basis=payload.get("bouncer_das_basis"),
            bouncer_das_explained=payload.get("bouncer_das_explained"),
            bouncer_das_singular_values=payload.get("bouncer_das_singular_values"),
            bouncer_das_weights=payload.get("bouncer_das_weights"),
            bouncer_das_target_scale=payload.get("bouncer_das_target_scale"),
            bouncer_das_capability_profile=payload.get("bouncer_das_capability_profile"),
            bouncer_das_capability_damage=payload.get("bouncer_das_capability_damage"),
            bouncer_das_capability_passed=payload.get("bouncer_das_capability_passed"),
        )


@dataclass
class LayerAtlas:
    layer_idx: int
    corporate_axis: torch.Tensor
    corporate_axis_clean: torch.Tensor
    refusal_axis: torch.Tensor
    angle_degrees: float
    neutral_midpoint_projection: float
    classifier_coef: torch.Tensor
    per_projection: Dict[str, ProjectionAtlas] = field(default_factory=dict)
    activation_histogram: Dict[str, torch.Tensor] = field(default_factory=dict)
    classifier_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": int(self.layer_idx),
            "corporate_axis": self.corporate_axis.detach().cpu(),
            "corporate_axis_clean": self.corporate_axis_clean.detach().cpu(),
            "refusal_axis": self.refusal_axis.detach().cpu(),
            "angle_degrees": float(self.angle_degrees),
            "neutral_midpoint_projection": float(self.neutral_midpoint_projection),
            "classifier_coef": self.classifier_coef.detach().cpu(),
            "per_projection": {k: v.to_dict() for k, v in self.per_projection.items()},
            "activation_histogram": {k: v.detach().cpu() for k, v in self.activation_histogram.items()},
            "classifier_accuracy": float(self.classifier_accuracy),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LayerAtlas":
        return cls(
            layer_idx=int(payload["layer_idx"]),
            corporate_axis=payload["corporate_axis"],
            corporate_axis_clean=payload["corporate_axis_clean"],
            refusal_axis=payload["refusal_axis"],
            angle_degrees=float(payload["angle_degrees"]),
            neutral_midpoint_projection=float(payload["neutral_midpoint_projection"]),
            classifier_coef=payload["classifier_coef"],
            per_projection={
                str(k): ProjectionAtlas.from_dict(v) for k, v in dict(payload.get("per_projection", {})).items()
            },
            activation_histogram={
                str(k): v for k, v in dict(payload.get("activation_histogram", {})).items()
            },
            classifier_accuracy=float(payload.get("classifier_accuracy", 0.0)),
        )


@dataclass
class BrainAtlas:
    model_name: str
    num_layers: int
    hidden_size: int
    sacred_layers: List[int]
    layers: Dict[int, LayerAtlas]
    probe_config: Dict[str, Any]
    built_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_layers": int(self.num_layers),
            "hidden_size": int(self.hidden_size),
            "sacred_layers": [int(x) for x in self.sacred_layers],
            "layers": {int(k): v.to_dict() for k, v in self.layers.items()},
            "probe_config": dict(self.probe_config),
            "built_at": self.built_at,
        }

    def save(self, path: str) -> None:
        torch.save(self.to_dict(), path)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BrainAtlas":
        return cls(
            model_name=str(payload["model_name"]),
            num_layers=int(payload["num_layers"]),
            hidden_size=int(payload["hidden_size"]),
            sacred_layers=[int(x) for x in payload.get("sacred_layers", [])],
            layers={int(k): LayerAtlas.from_dict(v) for k, v in dict(payload.get("layers", {})).items()},
            probe_config=dict(payload.get("probe_config", {})),
            built_at=str(payload.get("built_at") or datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def load(cls, path: str) -> "BrainAtlas":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"BrainAtlas payload must be dict, got {type(payload)!r}")
        return cls.from_dict(payload)
