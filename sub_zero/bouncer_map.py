"""Parse an atlas report JSON → bouncer_svs[layer][projection] dict.

Usage
-----
from sub_zero.bouncer_map import load_bouncer_svs

bouncer_svs = load_bouncer_svs(
    "atlas-gemma4-e2b-report.json",
    classifier_score_threshold=0.6,   # keep SVs with score ≥ this
    dark_variance_threshold=None,     # optional extra filter
    sacred_only=False,                # restrict to sacred layers only
    top_n_layers=None,                # restrict to top N layers by bouncer_pct
)

# bouncer_svs[layer_idx][proj_name] = frozenset of SV indices
# e.g. bouncer_svs[17]["gate_proj"] == frozenset({3, 7, 11})
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, FrozenSet, Optional


BouncerSVs = Dict[int, Dict[str, FrozenSet[int]]]


def load_bouncer_svs(
    report_path: str | Path,
    classifier_score_threshold: float = 0.5,
    dark_variance_threshold: Optional[float] = None,
    sacred_only: bool = False,
    top_n_layers: Optional[int] = None,
) -> BouncerSVs:
    """Parse atlas report JSON and return bouncer SV indices per layer/projection.

    Parameters
    ----------
    report_path:
        Path to the *-report.json file written by the modal probe app.
    classifier_score_threshold:
        Minimum classifier_score for an SV to be included. 0.5 keeps everything
        the probe already selected; raise to 0.7+ for only the sharpest directions.
    dark_variance_threshold:
        If set, also require dark_variance ≥ this value (filters for SVs that are
        active on dark/refusal-adjacent activations, not just classifier-salient).
    sacred_only:
        If True, only return bouncers in sacred (protected) layers.
    top_n_layers:
        If set, restrict to the top-N layers ranked by bouncer_pct.

    Returns
    -------
    dict[layer_idx, dict[proj_name, frozenset[sv_idx]]]
        Empty inner sets mean no SVs passed the threshold for that projection.
    """
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))

    sacred_set: set[int] = {int(x) for x in report.get("sacred_layers", [])}

    # Optionally restrict to top-N layers by bouncer_pct
    allowed_layers: Optional[set[int]] = None
    if top_n_layers is not None:
        ranked = sorted(report["layers"], key=lambda r: r["bouncer_pct"], reverse=True)
        allowed_layers = {int(r["layer"]) for r in ranked[:top_n_layers]}

    result: BouncerSVs = {}

    for layer_row in report["layers"]:
        li = int(layer_row["layer"])

        if sacred_only and li not in sacred_set:
            continue
        if allowed_layers is not None and li not in allowed_layers:
            continue

        proj_map: Dict[str, FrozenSet[int]] = {}
        for proj_row in layer_row["projections"]:
            pname = proj_row["projection"]
            kept: set[int] = set()
            for sv in proj_row.get("top_bouncer_svs", []):
                if sv["classifier_score"] < classifier_score_threshold:
                    continue
                if dark_variance_threshold is not None:
                    if sv.get("dark_variance", 0.0) < dark_variance_threshold:
                        continue
                kept.add(int(sv["sv_index"]))
            proj_map[pname] = frozenset(kept)

        if any(proj_map.values()):
            result[li] = proj_map

    return result


def summarise(bouncer_svs: BouncerSVs) -> str:
    """Return a human-readable summary string for logging."""
    lines = [f"bouncer_svs: {len(bouncer_svs)} layers"]
    total_svs = 0
    for li in sorted(bouncer_svs):
        proj_parts = []
        for pname, svs in sorted(bouncer_svs[li].items()):
            if svs:
                proj_parts.append(f"{pname}={sorted(svs)}")
                total_svs += len(svs)
        if proj_parts:
            lines.append(f"  L{li:>02}: " + "  ".join(proj_parts))
    lines.append(f"  total bouncer SVs: {total_svs}")
    return "\n".join(lines)
