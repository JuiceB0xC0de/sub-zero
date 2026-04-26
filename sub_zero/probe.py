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
    max_prompts_per_class: int = 32
    max_length: int = 256
    batch_size: int = 8
    classifier_accuracy_floor: float = 0.55
    bouncer_wanda_ratio: float = 1.8
    bouncer_composite_quantile: float = 0.85
    dark_variance_quantile: float = 0.50
    refusal_angle_degrees: float = 60.0
    sacred_top_k_percent: float = 0.50
    num_probe_batches: int = 5
    num_refusal_directions: int = 3
    layer_limit: Optional[int] = None
    coherence_pass: bool = True
    causal_validate: bool = True
    causal_validate_batch: int = 4
    causal_max_candidates: int = 20
    causal_keep_quantile: float = 0.5
    das_refine: bool = True
    das_target_rank: int = 2
    das_batch: int = 4
    das_explained_floor: float = 0.05
    das_min_scale: float = 0.15
    das_probe_token_ids: Optional[List[int]] = None  # if set, project deltas to these tokens before SVD
    # Gemma-style chat-template wrapping for corp/auth/neutral (model role) and red_team (user role)
    chat_template: bool = True
    chat_user_preamble: str = "respond."
    # Bouncer scope filters — applied at SVD time, propagates through all downstream gates
    skip_attention_projections: bool = True
    skip_embedding_layer: bool = True
    skip_unembedding_layer: bool = True
    skip_projections: Optional[List[str]] = None  # explicit override, else attention defaults
    skip_global_layers: Optional[List[int]] = None  # explicit override, else [0, n-1]


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


def _apply_chat_template(
    prompts: Sequence[str],
    role: str,
    user_preamble: str = "respond.",
) -> List[str]:
    """Wrap prompts with Gemma chat-template tokens.

    role='model':   prompt content sits inside the model turn (assistant continuation).
                    Activations at the last token represent 'model is producing this'.
    role='user':    prompt content is the user input; model turn is opened but empty.
                    Activations at the last token represent 'model about to respond to this'.

    The bouncer dimensions were RLHF-conditioned to fire at positions inside or
    immediately following the model-role prefix. Plain text bypasses that conditioning
    entirely, which is why pre-template baselines were 13+ nats per token.
    """
    out: List[str] = []
    for p in prompts:
        p = p.strip()
        if role == "model":
            out.append(
                f"<bos><start_of_turn>user\n{user_preamble}<end_of_turn>\n"
                f"<start_of_turn>model\n{p}"
            )
        elif role == "user":
            out.append(
                f"<bos><start_of_turn>user\n{p}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
        else:
            out.append(p)
    return out


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
    batch_size: int = 8,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
    device = model_device(model)
    hidden_size = int(getattr(model.config, "hidden_size", 0))

    layer_rows: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(layers))}
    proj_rows: Dict[int, Dict[str, List[torch.Tensor]]] = {
        i: {name: [] for name in get_projection_map(layer).keys()}
        for i, layer in enumerate(layers)
    }

    # Prefer left-padding on real HF tokenizers so the last real token of every
    # sample lands at position -1. Toy/minimal tokenizers may not support it —
    # fall back to attention_mask-based indexing for the true last token.
    orig_padding_side = getattr(tokenizer, "padding_side", None)
    use_left_pad = hasattr(tokenizer, "padding_side")
    if use_left_pad:
        try:
            tokenizer.padding_side = "left"
        except Exception:
            use_left_pad = False
    if (
        hasattr(tokenizer, "pad_token")
        and getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    try:
        for batch_start in range(0, len(prompts), max(1, int(batch_size))):
            batch = list(prompts[batch_start:batch_start + max(1, int(batch_size))])
            if not batch:
                continue

            # Encode the batch first so we know where each sample's last real
            # token lives (needed for the right-padded fallback path).
            enc = tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
            attention_mask = enc.get("attention_mask")

            if use_left_pad or attention_mask is None:
                def _last_slice(x: torch.Tensor) -> torch.Tensor:
                    return x[:, -1, :]
            else:
                last_idx = (attention_mask.sum(dim=1) - 1).clamp(min=0)
                def _last_slice(x: torch.Tensor, _li=last_idx) -> torch.Tensor:
                    bs = x.shape[0]
                    return x[torch.arange(bs, device=x.device), _li.to(x.device), :]

            proj_inputs: Dict[Tuple[int, str], torch.Tensor] = {}
            handles = []
            for li, layer in enumerate(layers):
                for pname, pmod in get_projection_map(layer).items():
                    def _mk(li=li, pname=pname, _last=_last_slice):
                        def _hook(_mod, inp, _out):
                            x = inp[0]
                            if isinstance(x, tuple):
                                x = x[0]
                            if isinstance(x, torch.Tensor) and x.ndim == 3:
                                proj_inputs[(li, pname)] = _last(x).detach().float().cpu()
                        return _hook
                    handles.append(pmod.register_forward_hook(_mk()))

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, use_cache=False)

            for h in handles:
                h.remove()

            hs = list(out.hidden_states or [])
            if not hs:
                continue
            for li in range(min(len(layers), len(hs) - 1)):
                # [batch, hidden] — one row per sample
                v = _last_slice(hs[li + 1]).detach().float().cpu()
                if hidden_size == 0:
                    hidden_size = int(v.shape[-1])
                for b in range(v.shape[0]):
                    layer_rows[li].append(v[b])
            for (li, pname), v in proj_inputs.items():
                for b in range(v.shape[0]):
                    proj_rows[li][pname].append(v[b])
    finally:
        if orig_padding_side is not None:
            try:
                tokenizer.padding_side = orig_padding_side
            except Exception:
                pass

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
        param_grads: Dict[Tuple[int, str], torch.Tensor] = {}
        hooks = []

        for li, layer in enumerate(layers):
            if li not in proj_svd:
                continue
            for pname, pmod in get_projection_map(layer).items():
                if pname not in proj_svd[li]:
                    continue
                def _mk(li=li, pname=pname, w=pmod.weight):
                    def _hook(g):
                        param_grads[(li, pname)] = g.detach().float().cpu()
                    hooks.append(w.register_hook(_hook))
                _mk()

        enc = tokenizer(corp_prompts[idx], return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        model.zero_grad()
        out = model(**enc, use_cache=False)
        logits = out.logits[0, :-1, :]
        targets = enc["input_ids"][0, 1:]
        loss = F.cross_entropy(logits, targets)
        loss.backward()

        for h in hooks:
            h.remove()

        for li in range(len(layers)):
            if li not in proj_svd:
                continue
            for pname in proj_svd[li]:
                g_w = param_grads.get((li, pname))
                if g_w is None:
                    continue
                
                u, s, vh = proj_svd[li][pname]
                c_act = corp_proj_acts[li].get(pname)
                a_act = auth_proj_acts[li].get(pname)
                
                if c_act is None or a_act is None or c_act.shape[-1] != vh.shape[1]:
                    continue
                
                c_sv = (c_act @ vh.T).mean(0)   # [rank]
                a_sv = (a_act @ vh.T).mean(0)   # [rank]
                diff = c_sv - a_sv
                
                # Project weight grad into right-singular space: g_sv[k] = ||g_w @ vh[k]||
                # g_w is [out_dim, in_dim], vh.T is [in_dim, rank]
                g_sv = (g_w @ vh.T).norm(dim=0) # [rank]
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
# Stage 5 – cross-layer coherence repass
# ---------------------------------------------------------------------------

def _knee_select(scores: torch.Tensor, max_frac: float = 0.30) -> List[int]:
    sorted_c, sort_idx = torch.sort(scores, descending=True)
    n = sorted_c.numel()
    if n < 4 or float(sorted_c[0] - sorted_c[-1]) <= 1e-6:
        return sort_idx[: max(1, int(0.10 * n))].tolist()
    xs = torch.linspace(0.0, 1.0, n)
    ys = (sorted_c - sorted_c[-1]) / (sorted_c[0] - sorted_c[-1] + 1e-12)
    dist = (ys + xs - 1.0).abs()
    k_cut = int(torch.argmax(dist).item()) + 1
    k_cut = max(1, min(k_cut, max(1, int(max_frac * n))))
    return sort_idx[:k_cut].tolist()


def _coherence_repass(
    atlas_layers: Dict[int, "LayerAtlas"],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    top_neighbor_frac: float = 0.30,
    neighbor_cap: int = 64,
) -> None:
    """Multiply each direction's composite by neighbor-layer coherence, then re-knee.

    Real bouncer pathways persist across consecutive layers. A right-singular
    direction at layer L should align with at least one high-composite direction
    in the same projection at L-1 or L+1. Lone hits get demoted; persistent
    pathways get amplified.
    """
    for li, layer_at in list(atlas_layers.items()):
        if li not in proj_svd:
            continue
        for pname, projat in layer_at.per_projection.items():
            if pname not in proj_svd[li]:
                continue
            _, _, vh_l = proj_svd[li][pname]
            composite = projat.per_direction_classifier_score
            if composite.numel() == 0:
                continue

            neighbor_dirs: List[torch.Tensor] = []
            for nb in (li - 1, li + 1):
                if nb not in atlas_layers or nb not in proj_svd:
                    continue
                nb_proj = atlas_layers[nb].per_projection.get(pname)
                if nb_proj is None or pname not in proj_svd[nb]:
                    continue
                _, _, vh_nb = proj_svd[nb][pname]
                if vh_nb.shape[1] != vh_l.shape[1]:
                    continue
                nb_score = nb_proj.per_direction_classifier_score
                n_top = min(int(top_neighbor_frac * nb_score.numel()), neighbor_cap, nb_score.numel())
                if n_top <= 0:
                    continue
                top_idx = torch.topk(nb_score, n_top).indices
                neighbor_dirs.append(vh_nb[top_idx])

            if not neighbor_dirs:
                continue

            nbm = torch.cat(neighbor_dirs, dim=0).float()         # [N, in_dim]
            sim = (vh_l.float() @ nbm.T).abs()                    # [rank, N]
            coh = sim.max(dim=1).values                           # [rank]
            multiplier = 0.5 + 0.5 * coh
            new_composite = composite * multiplier

            new_idx = _knee_select(new_composite)
            rank = projat.S.numel()
            new_scales = torch.ones(rank)
            for ki in new_idx:
                new_scales[ki] = 0.15

            projat.per_direction_classifier_score = new_composite
            projat.bouncer_sv_indices = torch.tensor(new_idx, dtype=torch.long)
            projat.per_direction_target_scale = new_scales
            print(
                f"  [coherence | L{li:>2} | {pname:<8}] "
                f"coh_max={float(coh.max()):.3f}  bouncers→{len(new_idx)}/{rank}"
            )


# ---------------------------------------------------------------------------
# Stage 6 – causal ablation gate
# ---------------------------------------------------------------------------

def _causal_validate(
    model: torch.nn.Module,
    tokenizer,
    layers: Sequence[torch.nn.Module],
    atlas_layers: Dict[int, "LayerAtlas"],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    corp_prompts: Sequence[str],
    auth_prompts: Sequence[str],
    max_length: int,
    batch: int = 4,
    max_candidates: int = 20,
    keep_quantile: float = 0.5,
) -> None:
    """Forward-pre-hook ablation: project candidate direction out of the projection
    input, then measure loss shift on corp vs auth. Real bouncers should:
      - INCREASE corp loss when ablated (direction was load-bearing for compliance)
      - DECREASE auth loss when ablated (direction was suppressing authentic voice)
    Score = (auth_clean - auth_abl) + (corp_abl - corp_clean). Positive = causal.
    """
    device = model_device(model)
    if (
        hasattr(tokenizer, "pad_token")
        and getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    def _enc(prompts):
        e = tokenizer(list(prompts), return_tensors="pt", truncation=True,
                      padding=True, max_length=max_length)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in e.items()}

    corp_enc = _enc(corp_prompts[: max(1, batch)])
    auth_enc = _enc(auth_prompts[: max(1, batch)])

    def _loss(enc):
        with torch.no_grad():
            out = model(**enc, use_cache=False)
        logits = out.logits[..., :-1, :].float()
        targets = enc["input_ids"][..., 1:]
        return float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        ).item())

    corp_clean = _loss(corp_enc)
    auth_clean = _loss(auth_enc)
    print(f"  [causal] baseline  corp_loss={corp_clean:.4f}  auth_loss={auth_clean:.4f}")

    for li, layer_at in list(atlas_layers.items()):
        if li not in proj_svd:
            continue
        for pname, projat in layer_at.per_projection.items():
            if pname not in proj_svd[li]:
                continue
            _, _, vh = proj_svd[li][pname]
            pmod_map = get_projection_map(layers[li])
            pmod = pmod_map.get(pname)
            if pmod is None:
                continue

            cand = projat.bouncer_sv_indices.tolist()
            if not cand:
                continue
            cand = cand[: max_candidates]

            scored: List[Tuple[int, float]] = []
            for sv_idx in cand:
                v_cpu = vh[sv_idx].float()

                def _pre_hook(_m, args, _v=v_cpu):
                    if not args:
                        return None
                    x = args[0]
                    if not isinstance(x, torch.Tensor):
                        return None
                    v_dt = _v.to(dtype=x.dtype, device=x.device)
                    coeff = x @ v_dt
                    proj = coeff.unsqueeze(-1) * v_dt
                    return (x - proj,) + tuple(args[1:])

                handle = pmod.register_forward_pre_hook(_pre_hook)
                try:
                    corp_abl = _loss(corp_enc)
                    auth_abl = _loss(auth_enc)
                finally:
                    handle.remove()

                score = (auth_clean - auth_abl) + (corp_abl - corp_clean)
                scored.append((int(sv_idx), float(score)))

            if not scored:
                continue
            arr = torch.tensor([s for _, s in scored])
            tau = max(0.0, float(torch.quantile(arr, keep_quantile)))
            kept = [sv for (sv, s) in scored if s > tau]
            if not kept:
                kept = [max(scored, key=lambda x: x[1])[0]]

            rank = projat.S.numel()
            new_scales = torch.ones(rank)
            for ki in kept:
                new_scales[ki] = 0.15
            projat.bouncer_sv_indices = torch.tensor(kept, dtype=torch.long)
            projat.per_direction_target_scale = new_scales

            print(
                f"  [causal | L{li:>2} | {pname:<8}] "
                f"{len(cand)} → {len(kept)} kept "
                f"(τ={tau:+.4f}, max={float(arr.max()):+.4f}, "
                f"min={float(arr.min()):+.4f})"
            )


# ---------------------------------------------------------------------------
# Stage 7 – DAS rotation gate (SVD of per-candidate logit deltas)
# ---------------------------------------------------------------------------

def _last_position_logits(model: torch.nn.Module, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return [B, vocab] logits at each sample's last real (non-pad) position."""
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    logits = out.logits.float()                                 # [B, S, V]
    am = enc.get("attention_mask")
    if am is None:
        return logits[:, -1, :]
    last_idx = (am.sum(dim=1) - 1).clamp(min=0).to(logits.device)
    bs = logits.shape[0]
    return logits[torch.arange(bs, device=logits.device), last_idx, :]


def _das_refine(
    model: torch.nn.Module,
    tokenizer,
    layers: Sequence[torch.nn.Module],
    atlas_layers: Dict[int, "LayerAtlas"],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    auth_prompts: Sequence[str],
    max_length: int,
    batch: int = 4,
    target_rank: int = 2,
    explained_floor: float = 0.05,
    min_scale: float = 0.15,
    probe_token_ids: Optional[List[int]] = None,
) -> None:
    """SVD of per-candidate logit-shift matrix. Finds non-axis-aligned causal axes
    within the surviving bouncer subspace.

    For each (layer, projection) with surviving candidates v_1..v_k:
      Δ_i = mean over auth batch of [logit_clean - logit_ablated_i]   ∈ R^vocab
      Δ   = stack(Δ_i)                                                ∈ R^[k, vocab]
      U Σ Vᵀ = svd(Δ)
      DAS basis = Uᵀ[:r] @ vh[bouncer_sv_indices]                     ∈ R^[r, in_dim]

    Σ²/Σ Σ² gives explained-causal-variance ratio. Drop axes below explained_floor.
    """
    if not auth_prompts:
        return
    device = model_device(model)
    if (
        hasattr(tokenizer, "pad_token")
        and getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        try:
            tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            pass

    enc = tokenizer(
        list(auth_prompts[: max(1, batch)]),
        return_tensors="pt", truncation=True, padding=True, max_length=max_length,
    )
    enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
    clean_logits = _last_position_logits(model, enc)             # [B, V]
    # Optional probe-token projection: shrinks vocab dim from ~256k → |probe|.
    # Use a curated refusal/compliance token set when memory matters.
    probe_idx = (
        torch.tensor(probe_token_ids, dtype=torch.long, device=clean_logits.device)
        if probe_token_ids else None
    )
    if probe_idx is not None:
        clean_logits = clean_logits.index_select(-1, probe_idx)

    for li, layer_at in list(atlas_layers.items()):
        if li not in proj_svd:
            continue
        for pname, projat in layer_at.per_projection.items():
            if pname not in proj_svd[li]:
                continue
            _, _, vh = proj_svd[li][pname]
            pmod = get_projection_map(layers[li]).get(pname)
            if pmod is None:
                continue

            cand = projat.bouncer_sv_indices.tolist()
            if len(cand) < 2:
                # nothing to rotate — single direction is its own basis
                if len(cand) == 1:
                    v = vh[cand[0]].float().unsqueeze(0)         # [1, in_dim]
                    projat.bouncer_das_basis = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    projat.bouncer_das_explained = torch.ones(1)
                    projat.bouncer_das_singular_values = torch.ones(1)
                    projat.bouncer_das_weights = torch.eye(1)
                    projat.bouncer_das_target_scale = torch.tensor([min_scale])
                continue

            deltas: List[torch.Tensor] = []
            for sv_idx in cand:
                v_cpu = vh[sv_idx].float()

                def _pre_hook(_m, args, _v=v_cpu):
                    if not args:
                        return None
                    x = args[0]
                    if not isinstance(x, torch.Tensor):
                        return None
                    v_dt = _v.to(dtype=x.dtype, device=x.device)
                    coeff = x @ v_dt
                    proj = coeff.unsqueeze(-1) * v_dt
                    return (x - proj,) + tuple(args[1:])

                handle = pmod.register_forward_pre_hook(_pre_hook)
                try:
                    abl_logits = _last_position_logits(model, enc)
                finally:
                    handle.remove()

                if probe_idx is not None:
                    abl_logits = abl_logits.index_select(-1, probe_idx)
                # mean across batch, sign convention: clean - ablated
                # (positive components = tokens whose logit DROPPED when bouncer ablated)
                delta = (clean_logits - abl_logits).mean(dim=0).cpu()
                deltas.append(delta)

            D = torch.stack(deltas, dim=0).float()               # [k, V]
            try:
                U, S, _ = torch.linalg.svd(D, full_matrices=False)
            except Exception as e:
                print(f"  [das | L{li:>2} | {pname:<8}] svd failed: {e}")
                continue

            total = float((S ** 2).sum().clamp(min=1e-12))
            explained = (S ** 2) / total                         # [min(k, V)]

            r_max = min(target_rank, U.shape[1])
            r = 0
            for j in range(r_max):
                if float(explained[j]) >= explained_floor:
                    r += 1
                else:
                    break
            r = max(1, r)

            W = U[:, :r]                                          # [k, r]
            B = vh[cand].float()                                  # [k, in_dim]
            das_basis = W.T @ B                                   # [r, in_dim]
            das_basis = das_basis / das_basis.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            # Per-axis attenuation: axes carrying more causal variance get
            # attenuated harder (toward min_scale). Trailing axes get gentler
            # treatment to preserve capability.  scale_r = 1 - (1 - min) * explained_r
            target_scale = (1.0 - (1.0 - min_scale) * explained[:r]).clamp(
                min=min_scale, max=1.0
            )

            projat.bouncer_das_basis = das_basis
            projat.bouncer_das_explained = explained[:r].clone()
            projat.bouncer_das_singular_values = S[:r].clone()
            projat.bouncer_das_weights = W.T.contiguous()         # [r, k]
            projat.bouncer_das_target_scale = target_scale

            exp_str = ", ".join(f"{float(e):.2%}" for e in explained[:r])
            scale_str = ", ".join(f"{float(s):.2f}" for s in target_scale)
            cum = float(explained[:r].sum())
            print(
                f"  [das | L{li:>2} | {pname:<8}] k={len(cand)} → r={r}  "
                f"explained=[{exp_str}]  cum={cum:.1%}  scales=[{scale_str}]"
            )


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

    if getattr(config, "chat_template", True):
        preamble = getattr(config, "chat_user_preamble", "respond.")
        print(f"[sub-zero] applying Gemma chat template (user_preamble={preamble!r}) ...")
        corp = _apply_chat_template(corp, role="model", user_preamble=preamble)
        auth = _apply_chat_template(auth, role="model", user_preamble=preamble)
        neu  = _apply_chat_template(neu,  role="model", user_preamble=preamble)
        red  = _apply_chat_template(red or [], role="user", user_preamble=preamble)

    layers   = resolve_layers(model)
    if config.layer_limit is not None:
        layers = layers[:config.layer_limit]
    n_layers = len(layers)

    # Resolve scope filters: which (layer, projection) cells to entirely exclude.
    skip_proj_set = set(
        config.skip_projections if config.skip_projections is not None
        else (["q_proj", "k_proj", "v_proj", "o_proj"]
              if getattr(config, "skip_attention_projections", True) else [])
    )
    skip_layer_set: set = set(config.skip_global_layers or [])
    if getattr(config, "skip_embedding_layer", True):
        skip_layer_set.add(0)
    if getattr(config, "skip_unembedding_layer", True):
        skip_layer_set.add(n_layers - 1)
    if skip_proj_set or skip_layer_set:
        print(
            f"[sub-zero] scope filter: skipping projections={sorted(skip_proj_set) or 'none'}, "
            f"global layers={sorted(skip_layer_set) or 'none'}"
        )

    print(f"[sub-zero] probing {n_layers} layers" + (f" (capped at {config.layer_limit})" if config.layer_limit else ""))
    print(f"[sub-zero] layer 17 module keys: {[n for n, _ in layers[min(17, n_layers-1)].named_modules()][:20]}")

    if task_batches:
        sacred_layers, _ = run_aletheia(
            model, task_batches=task_batches,
            num_probe_batches=config.num_probe_batches,
            top_k_percent=config.sacred_top_k_percent,
        )
    else:
        n_sel = max(1, int(round(n_layers * config.sacred_top_k_percent)))
        sacred_layers = list(range(n_layers - n_sel, n_layers))

    print(f"[sub-zero] stage 1/4: forward activation capture (batch_size={config.batch_size}) ...")
    corp_h, corp_p = _capture_forward(model, tokenizer, corp,       layers, config.max_length, config.batch_size)
    auth_h, auth_p = _capture_forward(model, tokenizer, auth,       layers, config.max_length, config.batch_size)
    neu_h,  neu_p  = _capture_forward(model, tokenizer, neu,        layers, config.max_length, config.batch_size)
    red_h,  _      = _capture_forward(model, tokenizer, red or neu, layers, config.max_length, config.batch_size)

    hidden_size = int(next(iter(corp_h.values())).shape[-1])

    print("[sub-zero] stage 2/4: SVD decomposition (GPU if available) ...")
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
    svd_device = model_device(model)
    for li in sacred_layers:
        if li in skip_layer_set:
            continue
        proj_svd[li] = {}
        for pname, pmod in get_projection_map(layers[li]).items():
            if pname in skip_proj_set:
                continue
            w = pmod.weight.detach()
            if w.ndim != 2 or min(w.shape) < 2:
                continue
            try:
                # fp32 for numerical stability; run on whatever device the model is on.
                # torch.linalg.svd on CUDA is 10-50x faster than CPU for these sizes.
                w_dev = w.to(device=svd_device, dtype=torch.float32)
                u_d, s_d, vh_d = torch.linalg.svd(w_dev, full_matrices=False)
                proj_svd[li][pname] = (u_d.cpu(), s_d.cpu(), vh_d.cpu())
                del w_dev, u_d, s_d, vh_d
            except Exception:
                continue

    print("[sub-zero] stage 3/4: AtP gradient probe ...")
    # neu_p is not used after stage 1; free it before enabling grads
    del neu_p
    torch.cuda.empty_cache()

    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    # AtP Smoke Test
    try:
        _test = tokenizer(corp[0], return_tensors="pt", truncation=True, max_length=config.max_length)
        _test = {k: v.to(model_device(model)) for k, v in _test.items()}
        model.zero_grad()
        _out = model(**_test, use_cache=False)
        F.cross_entropy(_out.logits[0, :-1], _test["input_ids"][0, 1:]).backward()
        _n = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
        print(f"[atp-smoke] {_n}/{sum(1 for _ in model.parameters())} params with grad")
        model.zero_grad()
    except Exception as e:
        print(f"[atp-smoke] FAILED: {e}")

    print(f"[sub-zero] model training mode: {model.training}")
    print(f"[sub-zero] proj_svd layers: {list(proj_svd.keys())}")
    print(f"[sub-zero] corp prompts for atp: {len(corp)}, auth: {len(auth)}")

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

            cone_in = cone_dirs  # [k, hidden_size]
            # Pick the correct geometric space for this projection: vh acts on input
            # (in_dim), u acts on output (out_dim). Match cone to the right one.
            if vh.shape[1] == cone_in.shape[1]:
                basis_target = vh.T                       # [in_dim, rank]
            elif u.shape[0] == cone_in.shape[1]:
                basis_target = u                          # [out_dim, rank]
            else:
                d = vh.shape[1]
                cone_in = cone_in[:, :d] if cone_in.shape[1] > d else F.pad(cone_in, (0, d - cone_in.shape[1]))
                basis_target = vh.T
            # Subspace projection: orthonormalize cone via QR, then per-SV-direction
            # alignment = ||Q @ v_k|| (norm of projection onto cone span), not max
            # cosine to any single centroid. Treats refusal as the subspace it is.
            try:
                Q, _ = torch.linalg.qr(cone_in.T.float())  # Q: [hidden, k']
                proj = Q.T @ basis_target.float()          # [k', rank]
                align = proj.pow(2).sum(0).sqrt().clamp(max=1.0)  # [rank]
            except Exception:
                align = (cone_in.float() @ basis_target.float()).abs().max(0).values

            def _norm01(t):
                lo, hi = t.min(), t.max()
                return (t - lo) / (hi - lo + 1e-12)

            wanda_ratio = (wanda_corp + 1e-12) / (wanda_auth + 1e-12)
            atp_n   = _norm01(atp.abs())
            align_n = _norm01(align)

            composite = (
                0.45 * _norm01(wanda_ratio)
                + 0.35 * atp_n
                + 0.20 * align_n
            )
            # Adaptive: knee-point on sorted composite (max distance from chord
            # connecting first to last point). Falls back to fixed quantile when
            # the curve is too flat to have a meaningful knee.
            sorted_c, sort_idx = torch.sort(composite, descending=True)
            n = sorted_c.numel()
            if n >= 4 and float(sorted_c[0] - sorted_c[-1]) > 1e-6:
                xs = torch.linspace(0.0, 1.0, n)
                ys = (sorted_c - sorted_c[-1]) / (sorted_c[0] - sorted_c[-1] + 1e-12)
                # distance from each (xs[i], ys[i]) to the chord y = 1 - x
                dist = (ys + xs - 1.0).abs()
                k_cut = int(torch.argmax(dist).item()) + 1
                # Guard rails: never accept more than 30% or fewer than 1
                k_cut = max(1, min(k_cut, max(1, int(0.30 * n))))
                bouncer_threshold = float(sorted_c[k_cut - 1])
                bouncer_idx = sort_idx[:k_cut].tolist()
            else:
                bouncer_threshold = float(torch.quantile(composite, getattr(config, "bouncer_composite_quantile", 0.85)))
                bouncer_idx = [ki for ki in range(rank) if float(composite[ki]) > bouncer_threshold]
            scales = torch.ones(rank)
            for ki in bouncer_idx:
                scales[ki] = 0.15

            # --- DIAGNOSTIC: raw pre-normalization distributions -----------
            # Purpose: verify whether the signal has meaningful structure
            # before the _norm01 + quantile gate compresses it to fixed 15%.
            # Keep until selection logic is calibrated, then remove.
            raw_wr = wanda_ratio
            raw_atp_abs = atp.abs()
            raw_align_vec = align
            def _q(t, q):
                return float(torch.quantile(t.float(), q)) if t.numel() else 0.0
            print(
                f"    [RAW | L{li:>2} | {pname:<9}] "
                f"wr[min={_q(raw_wr, 0.0):.2f} med={_q(raw_wr, 0.5):.2f} "
                f"p90={_q(raw_wr, 0.9):.2f} max={_q(raw_wr, 1.0):.2f}]  "
                f"atp[med={_q(raw_atp_abs, 0.5):.2e} p90={_q(raw_atp_abs, 0.9):.2e} "
                f"max={_q(raw_atp_abs, 1.0):.2e}]  "
                f"align[med={_q(raw_align_vec, 0.5):.3f} p90={_q(raw_align_vec, 0.9):.3f} "
                f"max={_q(raw_align_vec, 1.0):.3f}]  "
                f"composite[thresh={bouncer_threshold:.3f} "
                f"max={float(composite.max()):.3f} "
                f"unique={int(composite.unique().numel())}/{rank}]"
            )
            # ---------------------------------------------------------------

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
                per_direction_classifier_score=composite,
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

    if getattr(config, "coherence_pass", True):
        print("[sub-zero] stage 5/6: cross-layer coherence repass ...")
        _coherence_repass(atlas_layers, proj_svd)

    if getattr(config, "causal_validate", True):
        print("[sub-zero] stage 6/7: causal ablation gate ...")
        _causal_validate(
            model, tokenizer, layers, atlas_layers, proj_svd,
            corp_prompts=corp, auth_prompts=auth,
            max_length=config.max_length,
            batch=getattr(config, "causal_validate_batch", 4),
            max_candidates=getattr(config, "causal_max_candidates", 20),
            keep_quantile=getattr(config, "causal_keep_quantile", 0.5),
        )

    if getattr(config, "das_refine", True):
        print("[sub-zero] stage 7/7: DAS rotation gate ...")
        _das_refine(
            model, tokenizer, layers, atlas_layers, proj_svd,
            auth_prompts=auth,
            max_length=config.max_length,
            batch=getattr(config, "das_batch", 4),
            target_rank=getattr(config, "das_target_rank", 2),
            explained_floor=getattr(config, "das_explained_floor", 0.05),
            min_scale=getattr(config, "das_min_scale", 0.15),
            probe_token_ids=getattr(config, "das_probe_token_ids", None),
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