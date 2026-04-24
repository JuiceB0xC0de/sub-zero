from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .aletheia import run_aletheia
from .atlas import BrainAtlas, LayerAtlas, ProjectionAtlas
from .classifier import fit_corporate_axis
from .model_utils import get_projection_map, model_device, resolve_layers
from .propagation import trace_origin_layers


@dataclass
class ProbeConfig:
    corpora_dir: str
    corporate_file: str = "corporate_stems.txt"
    neutral_file: str = "neutral_stems.txt"
    authentic_file: str = "authentic_bella_samples.txt"
    red_team_file: str = "red_team_stems.txt"
    max_prompts_per_class: int = 64
    max_length: int = 256
    classifier_accuracy_floor: float = 0.55
    bouncer_wanda_ratio: float = 1.8
    bouncer_atp_quantile: float = 0.75
    bouncer_align_quantile: float = 0.60
    dark_variance_quantile: float = 0.50
    refusal_angle_degrees: float = 60.0
    sacred_top_k_percent: float = 0.50
    num_probe_batches: int = 5
    num_refusal_directions: int = 3


def _read_lines(path: Path, max_items: int) -> List[str]:
    if not path.exists():
        return []
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return lines[:max_items]


def _stack_or_empty(rows: List[torch.Tensor], dim: int) -> torch.Tensor:
    if rows:
        return torch.stack(rows, dim=0)
    return torch.empty(0, dim)


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    return v / n if float(n) > 1e-12 else torch.zeros_like(v)


def _unit_rows(v: torch.Tensor) -> torch.Tensor:
    """Normalize each row to unit length. Works on 1D or 2D."""
    if v.ndim == 1:
        n = v.norm()
        return v / n if float(n) > 1e-12 else torch.zeros_like(v)
    norms = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return v / norms


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    dot = float(torch.clamp(torch.dot(_unit(a), _unit(b)), -1.0, 1.0))
    return float(torch.rad2deg(torch.arccos(torch.tensor(dot))).item())


def _hist3(t: torch.Tensor) -> torch.Tensor:
    if t.numel() == 0:
        return torch.zeros(3)
    return torch.tensor([
        float((t < -0.2).float().sum()),
        float(((t >= -0.2) & (t <= 0.2)).float().sum()),
        float((t > 0.2).float().sum()),
    ])


# ---------------------------------------------------------------------------
# Stage 1 – pure forward: residual activations + projection INPUT activations
# ---------------------------------------------------------------------------

def _capture_forward(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[torch.nn.Module],
    max_length: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
    device = model_device(model)
    hidden_size = int(getattr(model.config, "hidden_size", 0))

    layer_rows: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(layers))}
    proj_rows: Dict[int, Dict[str, List[torch.Tensor]]] = {
        i: {name: [] for name in get_projection_map(layer).keys()}
        for i, layer in enumerate(layers)
    }

    for prompt in prompts:
        proj_inputs: Dict[Tuple[int, str], torch.Tensor] = {}
        handles = []
        for li, layer in enumerate(layers):
            for pname, pmod in get_projection_map(layer).items():
                def _mk(li=li, pname=pname):
                    def _hook(_mod, inp, _out):
                        x = inp[0]
                        if isinstance(x, tuple):
                            x = x[0]
                        if isinstance(x, torch.Tensor):
                            proj_inputs[(li, pname)] = x[0, -1, :].detach().float().cpu()
                    return _hook
                handles.append(pmod.register_forward_hook(_mk()))

        with torch.no_grad():
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, output_hidden_states=True, use_cache=False)

        for h in handles:
            h.remove()

        hs = list(out.hidden_states or [])
        if not hs:
            continue
        for li in range(min(len(layers), len(hs) - 1)):
            v = hs[li + 1][0, -1, :].detach().float().cpu()
            if hidden_size == 0:
                hidden_size = int(v.numel())
            layer_rows[li].append(v)
        for (li, pname), v in proj_inputs.items():
            proj_rows[li][pname].append(v)

    layer_out = {li: _stack_or_empty(rows, hidden_size) for li, rows in layer_rows.items()}
    proj_out: Dict[int, Dict[str, torch.Tensor]] = {}
    for li, m in proj_rows.items():
        proj_out[li] = {
            pname: _stack_or_empty(rows, rows[0].numel() if rows else 0)
            for pname, rows in m.items()
        }
    return layer_out, proj_out


# ---------------------------------------------------------------------------
# Stage 2 – backward: AtP gradient scores per singular direction
# ---------------------------------------------------------------------------

def _capture_atp_gradients(
    model: torch.nn.Module,
    tokenizer,
    corp_prompts: Sequence[str],
    auth_prompts: Sequence[str],
    layers: Sequence[torch.nn.Module],
    corp_proj_acts: Dict[int, Dict[str, torch.Tensor]],
    auth_proj_acts: Dict[int, Dict[str, torch.Tensor]],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    max_length: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    device = model_device(model)
    atp_accum: Dict[int, Dict[str, List[torch.Tensor]]] = {
        li: {pname: [] for pname in proj_svd.get(li, {})}
        for li in range(len(layers))
    }

    n_pairs = min(len(corp_prompts), len(auth_prompts))
    for idx in range(n_pairs):
        grad_store: Dict[Tuple[int, str], torch.Tensor] = {}
        fwd_store: Dict[Tuple[int, str], torch.Tensor] = {}
        handles = []

        for li, layer in enumerate(layers):
            if li not in proj_svd:
                continue
            for pname, pmod in get_projection_map(layer).items():
                if pname not in proj_svd[li]:
                    continue
                def _mk_fwd(li=li, pname=pname):
                    def _hook(_mod, inp, out):
                        act = out[0] if isinstance(out, tuple) else out
                        fwd_store[(li, pname)] = act[0, -1, :].detach().float().cpu()
                        act.retain_grad()
                        def _grad_hook(g):
                            grad_store[(li, pname)] = g[0, -1, :].detach().float().cpu()
                        act.register_hook(_grad_hook)
                    return _hook
                handles.append(pmod.register_forward_hook(_mk_fwd()))

        enc = tokenizer(corp_prompts[idx], return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        model.zero_grad()
        out = model(**enc, use_cache=False)
        logits = out.logits[0, :-1, :]
        targets = enc["input_ids"][0, 1:]
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        for h in handles:
            h.remove()

        for li in range(len(layers)):
            if li not in proj_svd:
                continue
            for pname in proj_svd[li]:
                if (li, pname) not in grad_store:
                    continue
                g = grad_store[(li, pname)]
                u, _, vh = proj_svd[li][pname]
                c_act = corp_proj_acts[li].get(pname)
                a_act = auth_proj_acts[li].get(pname)
                if c_act is None or a_act is None:
                    continue
                if c_act.shape[-1] == vh.shape[1]:
                    c_sv = (c_act @ vh.T).mean(0)
                    a_sv = (a_act @ vh.T).mean(0)
                else:
                    continue
                diff = c_sv - a_sv
                if g.shape[0] == u.shape[0]:
                    g_sv = u.T @ g
                else:
                    g_sv = torch.ones(diff.shape[0])
                atp_accum[li][pname].append(diff * g_sv)

    atp_out: Dict[int, Dict[str, torch.Tensor]] = {}
    for li in atp_accum:
        atp_out[li] = {}
        for pname, scores in atp_accum[li].items():
            if scores:
                atp_out[li][pname] = torch.stack(scores).mean(0)
            else:
                atp_out[li][pname] = torch.zeros(1)
    return atp_out


# ---------------------------------------------------------------------------
# Stage 3 – refusal concept cone (K-means over per-sample diff vectors)
# ---------------------------------------------------------------------------

def _compute_refusal_cone(
    corp_h: Dict[int, torch.Tensor],
    auth_h: Dict[int, torch.Tensor],
    k: int = 3,
) -> Dict[int, torch.Tensor]:
    cone: Dict[int, torch.Tensor] = {}
    for li in corp_h:
        c = corp_h[li]
        a = auth_h[li]
        n = min(c.shape[0], a.shape[0])
        if n < 2:
            cone[li] = torch.zeros(k, c.shape[-1])
            continue
        diffs = _unit_rows((c[:n] - a[:n]).float())     # [n, hidden] — per-row unit norm
        if n < k:
            mean_d = _unit_rows(diffs.mean(0, keepdim=True))
            cone[li] = mean_d.expand(k, -1).contiguous()
            continue
        idx = torch.randperm(n)[:k]
        centroids = diffs[idx].clone()
        for _ in range(5):
            sims = diffs @ centroids.T                  # [n, k]
            assign = sims.argmax(dim=1)                 # [n]
            for j in range(k):
                members = diffs[assign == j]
                if members.shape[0] > 0:
                    centroids[j] = _unit_rows(members.mean(0))
        cone[li] = centroids                            # [k, hidden]
    return cone


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_atlas(
    model: torch.nn.Module,
    tokenizer,
    config: ProbeConfig,
    task_batches=None,
    cache_path: Optional[str] = None,
) -> BrainAtlas:
    if cache_path and Path(cache_path).exists():
        return BrainAtlas.load(cache_path)

    corpora = Path(config.corpora_dir)
    corp = _read_lines(corpora / config.corporate_file,  config.max_prompts_per_class)
    neu  = _read_lines(corpora / config.neutral_file,    config.max_prompts_per_class)
    auth = _read_lines(corpora / config.authentic_file,  config.max_prompts_per_class)
    red  = _read_lines(corpora / config.red_team_file,   config.max_prompts_per_class)

    if not corp or not auth:
        raise RuntimeError("Need at least corporate + authentic corpus files.")
    if not neu:
        neu = auth

    layers   = resolve_layers(model)
    n_layers = len(layers)

    if task_batches:
        sacred_layers, _ = run_aletheia(
            model, task_batches=task_batches,
            num_probe_batches=config.num_probe_batches,
            top_k_percent=config.sacred_top_k_percent,
        )
    else:
        n_sel = max(1, int(round(n_layers * config.sacred_top_k_percent)))
        sacred_layers = list(range(n_layers - n_sel, n_layers))

    print("[sub-zero] stage 1/4: forward activation capture ...")
    corp_h, corp_p = _capture_forward(model, tokenizer, corp,       layers, config.max_length)
    auth_h, auth_p = _capture_forward(model, tokenizer, auth,       layers, config.max_length)
    neu_h,  neu_p  = _capture_forward(model, tokenizer, neu,        layers, config.max_length)
    red_h,  _      = _capture_forward(model, tokenizer, red or neu, layers, config.max_length)

    hidden_size = int(next(iter(corp_h.values())).shape[-1])

    print("[sub-zero] stage 2/4: SVD decomposition ...")
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    for li in sacred_layers:
        proj_svd[li] = {}
        for pname, pmod in get_projection_map(layers[li]).items():
            w = pmod.weight.detach().float().cpu()
            if w.ndim != 2 or min(w.shape) < 2:
                continue
            try:
                u, s, vh = torch.linalg.svd(w, full_matrices=False)
                proj_svd[li][pname] = (u, s, vh)
            except Exception:
                continue

    print("[sub-zero] stage 3/4: AtP gradient probe ...")
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    atp_scores = _capture_atp_gradients(
        model, tokenizer, corp, auth, layers,
        corp_p, auth_p, proj_svd, config.max_length,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[sub-zero] stage 4/4: refusal concept cone + bouncer scoring ...")
    refusal_cone = _compute_refusal_cone(corp_h, auth_h, k=config.num_refusal_directions)

    atlas_layers: Dict[int, LayerAtlas] = {}

    for li in range(n_layers):
        c, n_act, a, r = corp_h[li], neu_h[li], auth_h[li], red_h[li]
        if min(c.shape[0], n_act.shape[0], a.shape[0]) < 2:
            continue

        fit = fit_corporate_axis(c, n_act, a)
        refusal_axis = _unit((r.mean(0) - n_act.mean(0)).float()) if r.numel() else torch.zeros_like(fit.corporate_axis)
        angle = _angle_deg(fit.corporate_axis, refusal_axis)

        corp_clean = fit.corporate_axis.clone()
        if angle < config.refusal_angle_degrees and refusal_axis.norm() > 0:
            corp_clean = _unit(corp_clean - torch.dot(corp_clean, refusal_axis) * refusal_axis)

        class_hist = {
            "corporate": _hist3((c    @ corp_clean) - fit.neutral_midpoint_projection),
            "neutral":   _hist3((n_act @ corp_clean) - fit.neutral_midpoint_projection),
            "authentic": _hist3((a    @ corp_clean) - fit.neutral_midpoint_projection),
        }

        if li not in proj_svd:
            atlas_layers[li] = LayerAtlas(
                layer_idx=li, corporate_axis=fit.corporate_axis,
                corporate_axis_clean=corp_clean, refusal_axis=refusal_axis,
                angle_degrees=angle, neutral_midpoint_projection=fit.neutral_midpoint_projection,
                classifier_coef=fit.classifier_coef, per_projection={},
                activation_histogram=class_hist, classifier_accuracy=fit.classifier_accuracy,
            )
            continue

        cone_dirs = refusal_cone.get(li, torch.zeros(config.num_refusal_directions, hidden_size))

        per_projection: Dict[str, ProjectionAtlas] = {}

        for pname, (u, s, vh) in proj_svd[li].items():
            rank = s.numel()

            def _wanda(proj_acts, sv_mat):
                if proj_acts is None or proj_acts.numel() == 0:
                    return torch.zeros(rank)
                if proj_acts.shape[-1] == sv_mat.shape[1]:
                    energy = (proj_acts @ sv_mat.T).abs().mean(0)
                elif proj_acts.shape[-1] == sv_mat.shape[0]:
                    energy = (proj_acts @ sv_mat).abs().mean(0)
                else:
                    return torch.zeros(rank)
                return s.abs() * energy[:rank]

            wanda_corp = _wanda(corp_p[li].get(pname), vh)
            wanda_auth = _wanda(auth_p[li].get(pname), vh)

            auth_proj = auth_p[li].get(pname)
            if auth_proj is not None and auth_proj.numel() > 0 and auth_proj.shape[-1] == vh.shape[1]:
                dark_var = torch.var(auth_proj @ vh.T, dim=0)
            else:
                dark_var = torch.zeros(rank)

            atp = atp_scores.get(li, {}).get(pname, torch.zeros(rank))
            if atp.shape[0] != rank:
                atp = torch.zeros(rank)

            if u.shape[0] == cone_dirs.shape[1]:
                align = (cone_dirs @ u).abs().max(0).values
            elif vh.shape[1] == cone_dirs.shape[1]:
                align = (cone_dirs @ vh.T).abs().max(0).values
            else:
                align = torch.zeros(rank)

            def _norm01(t):
                lo, hi = t.min(), t.max()
                return (t - lo) / (hi - lo + 1e-12)

            wanda_ratio = (wanda_corp + 1e-12) / (wanda_auth + 1e-12)
            atp_n   = _norm01(atp.abs())
            align_n = _norm01(align)

            ratio_thresh = config.bouncer_wanda_ratio
            atp_thresh   = float(torch.quantile(atp_n,   config.bouncer_atp_quantile))
            align_thresh = float(torch.quantile(align_n, config.bouncer_align_quantile))

            cls_scores = 0.4 * _norm01(wanda_ratio) + 0.3 * atp_n + 0.3 * align_n

            bouncer_idx = []
            scales = torch.ones(rank)
            for ki in range(rank):
                if (
                    float(wanda_ratio[ki]) > ratio_thresh
                    and float(atp_n[ki])   > atp_thresh
                    and float(align_n[ki]) > align_thresh
                ):
                    bouncer_idx.append(ki)
                    scales[ki] = 0.15

            proj_dict = {
                off: corp_h[off] @ corp_clean
                for off in (li - 1, li)
                if off in corp_h and corp_h[off].numel()
            }
            origins = trace_origin_layers(proj_dict, [li])

            per_projection[pname] = ProjectionAtlas(
                proj_name=pname,
                S=s,
                bouncer_sv_indices=torch.tensor(bouncer_idx, dtype=torch.long),
                per_direction_classifier_score=cls_scores,
                per_direction_wanda_score=wanda_corp,
                per_direction_dark_variance=dark_var,
                per_direction_target_scale=scales,
                origin_layer={int(ki): int(origins.get(li, li)) for ki in bouncer_idx},
            )

            if bouncer_idx:
                print(
                    f"  [layer {li:>2} | {pname:<8}] "
                    f"bouncers={len(bouncer_idx)}/{rank}  "
                    f"wanda_ratio_max={float(wanda_ratio.max()):.2f}  "
                    f"atp_max={float(atp_n.max()):.3f}  "
                    f"align_max={float(align_n.max()):.3f}"
                )

        atlas_layers[li] = LayerAtlas(
            layer_idx=li,
            corporate_axis=fit.corporate_axis,
            corporate_axis_clean=corp_clean,
            refusal_axis=refusal_axis,
            angle_degrees=angle,
            neutral_midpoint_projection=fit.neutral_midpoint_projection,
            classifier_coef=fit.classifier_coef,
            per_projection=per_projection,
            activation_histogram=class_hist,
            classifier_accuracy=fit.classifier_accuracy,
        )

    atlas = BrainAtlas(
        model_name=str(getattr(model.config, "_name_or_path", "unknown")),
        num_layers=n_layers,
        hidden_size=hidden_size,
        sacred_layers=sorted(set(sacred_layers)),
        layers=atlas_layers,
        probe_config=asdict(config),
        built_at=datetime.now(timezone.utc).isoformat(),
    )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        atlas.save(cache_path)

    return atlas