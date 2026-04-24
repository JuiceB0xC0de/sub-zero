# Sub-Zero Design Specification

**Date:** 2026-04-24
**Author:** Claude (Opus 4.7) + Rick Holmberg
**Status:** Draft for user review
**Project location:** `/Users/chiggy/sub-zero/`
**Target models:** `google/gemma-3-2b-it` (perfection loop) → `google/gemma-3-4b-it` (scale-up)

---

## 1. Problem & Goal

Gemma-3 instruction-tuned checkpoints carry a baked-in corporate RLHF voice —
"As an AI model I'm here to help," "Happy to assist," "I'm a helpful
assistant" — that resists fine-tuning. Adapter-based methods (LoRA) leave
the corporate signal intact in the base weights; the adapter learns to work
around it rather than remove it. The model ends up sounding like an adapter
trying to hide a corporate voice underneath.

**Sub-Zero's goal:** identify the specific hidden dimensions in the base
weights that encode the corporate voice, physically attenuate them in place,
lock them with gradient hooks during training, and full-fine-tune everything
else. Scalpel on base-weight tissue — not a blanket on top of it.

**Non-goal:** suppressing safety refusals. Genuine refusals are
personality-preserving and explicitly preserved. Sub-Zero distinguishes the
corporate-compliance direction from the safety-refusal direction and
orthogonalizes the former against the latter before any suppression.

**Shippable artifact:** a **Brain Atlas** — a per-layer map of every Gemma
hidden dimension's role on a corporate↔authentic voice axis, for both
`gemma-3-2b-it` and `gemma-3-4b-it`, with a visualization notebook. The
community version of this deliverable is the atlas itself, usable as a
reference map for bigger Gemma models.

---

## 2. Scope & Sequencing

**Phase B (this spec, now):** standalone Python package `sub_zero/` at
`/Users/chiggy/sub-zero/`. Plugs into the existing Bella chaos trainer
alongside `DeepChaosScheduler` from `lucky-pick-scheduler`. Probe runs once
pre-training; applicator mutates weights once pre-training; DeepChaos
operates unchanged on non-sacred layers during training.

**Phase C (later, after e4b validation):** fold `sub_zero` into
`lucky_pick_scheduler` as a submodule with matching interface conventions
(`.from_model()`, `.step()`, `.remove()`). This spec writes code that will
port cleanly.

**Model sequencing:** perfect on `gemma-3-2b-it` first. Same architecture
family as `gemma-3-4b-it` — same `q_norm` / `k_norm` / `v_norm` RMSNorm
collapse risk, same `num_kv_shared_layers`. What works on 2B transfers. Port
to 4B once probe, applicator, and metrics are stable. Ship both atlases.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  PHASE 0 — BRAIN ATLAS (one-time, pre-training)                    │
│  Module:  sub_zero.probe                                            │
│  Inputs:  triplet corpus (corporate / neutral / authentic)          │
│           + red-team adversarial corpus                             │
│  Outputs: BrainAtlas  (sub_zero.atlas.BrainAtlas dataclass)        │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 1 — ALETHEIA LAYER SELECTION (one-time, pre-training)        │
│  Module:  sub_zero.aletheia                                         │
│  Inputs:  task batch (Bella training data sample)                   │
│  Outputs: sacred_layers: List[int]  (top-k by gradient norm)       │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 2 — SUB-ZERO APPLICATION (one-shot, pre-training)            │
│  Module:  sub_zero.applicator                                       │
│  Inputs:  model, BrainAtlas, sacred_layers                          │
│  Effects: in-place SVD scale of bouncer singular dirs on sacred     │
│           layers; registers grad hooks that zero bouncer-dim grads  │
│           during training; stores originals for restore()           │
├─────────────────────────────────────────────────────────────────────┤
│  PHASE 3 — TRAIN (existing Bella trainer, unchanged flow)           │
│  DeepChaosScheduler operates on non-sacred layers as today.         │
│  Optimizer updates non-bouncer directions on sacred layers.         │
│  W&B receives per-step metrics from sub_zero.instrumentation.       │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.1 Module Boundaries

Each module has one clear purpose, a well-defined interface, and can be
understood and tested independently.

| Module | Mutates model? | Requires grad? | Depends on |
|---|---|---|---|
| `atlas.py` | no | no | — (pure dataclass + io) |
| `probe.py` | no | no | `atlas`, `classifier`, `propagation` |
| `aletheia.py` | no | yes | — |
| `classifier.py` | no | no | sklearn |
| `propagation.py` | no | no | `atlas` |
| `hooks.py` | no (registers) | yes | — |
| `applicator.py` | **yes** | yes | `atlas`, `hooks`, `lucky_pick_scheduler.deep_chaos` (for layer resolution + quirk detection) |
| `instrumentation.py` | no | no | `atlas` |

The applicator is the only module that touches model weights. Everything
else is pure read or pure register-hook.

---

## 4. Phase 0 — The Probe (Brain Atlas Build)

### 4.1 Triplet Corpus

Three probe classes. Each class is ~50–100 short prompts or prompt-completion
stems. All three live under `sub-zero/corpora/` as plaintext files.

1. **Corporate-max** (`corporate_stems.txt`)
   - Stems that elicit maximum RLHF-assistant voice when continued by a
     chat-template-aware instruct model.
   - Vocabulary sourced from the semantic-word wordlist Rick is curating
     via Gemini: "assistant", "helpful", "AI model", "here to help",
     "happy to assist", etc.
   - Example stem: `"As a helpful AI assistant, I'm here to"`.
   - Run through the target model with the chat template engaged.

2. **Neutral** (`neutral_stems.txt`)
   - Factual-continuation text with **no chat template**. No persona
     engaged. Wikipedia-style sentences interrupted mid-thought.
   - Example stem: `"The capital of France is Paris, which was founded"`.
   - Run through the raw base model (completion mode, not chat mode).

3. **Authentic-max** (`authentic_bella_samples.jsonl`)
   - Best Bella-voice samples from Rick's existing training data. Voice,
     cadence, vocabulary characteristic of the target Bella personality.
   - Run through the target model with a minimal Bella-voice prefix.

4. **Red-team adversarial** (`red_team_stems.txt`)
   - Prompts that would elicit refusal from an RLHF-aligned model.
   - Maps the refusal direction. **Never suppressed.** Used as an
     orthogonal constraint during Phase 2 scaling.
   - Examples: standard refusal-evoking prompts from the prior art
     (Arditi et al. 2024 style).

### 4.2 Activation Capture

For each class, for each prompt:
- Forward-pass the model with `output_hidden_states=True`.
- Capture the residual stream at every layer at the **last token position**.
  (Using last-token aligns with prior refusal-direction work and avoids
  positional noise from averaging across variable-length sequences.)
- Also capture, on sacred layers, the pre-projection input activation at
  every `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`,
  `down_proj` — needed for Wanda scoring in 4.5.

Store as:
```python
activations[class_name][layer_idx] -> Tensor[n_prompts, hidden_dim]
proj_acts[layer_idx][proj_name]    -> Tensor[n_prompts, in_dim]
```

All captures via forward hooks. No gradients needed for this phase.

### 4.3 Per-Layer Linear Probe Classifier

Per layer `l`, train a logistic regression on the residual-stream activations
using only the triplet-corpus samples (not red-team):

```
X_train = concat(corp_acts[l], neutral_acts[l], auth_acts[l])   # [3N, hidden]
y_train = [+1]*N + [0]*N + [-1]*N                                # ordinal encoding
         # but trained as multinomial; we extract the corp-vs-auth axis

clf = LogisticRegression(multi_class="multinomial", C=1.0, max_iter=1000)
clf.fit(X_train, y_train)
```

Extract from the trained classifier:
- `corporate_axis[l]` = unit vector along `clf.coef_[corp_class] - clf.coef_[auth_class]` — the continuous axis from authentic (negative) through neutral (zero) to corporate (positive).
- `neutral_midpoint[l]` = the bias offset where that axis crosses zero.
- `corporate_magnitude(v) = dot(v - neutral_midpoint[l], corporate_axis[l])` — a scalar scoring function for any new activation `v`.

The classifier hyperplane IS the corporate-voice axis. Every future
dimension analysis projects onto it and gets a scalar position.

### 4.4 Refusal Direction (Probe B)

For each layer `l`:
- Mean red-team activation minus mean neutral activation = refusal vector.
- Unit-normalize: `refusal_axis[l]`.

Compute `angle[l] = arccos(dot(corporate_axis[l], refusal_axis[l]))`.

If `angle[l] < 60°` for any sacred layer, the two directions share subspace
and we **orthogonalize**:
```python
corp_clean[l] = corporate_axis[l] - (corporate_axis[l] · refusal_axis[l]) * refusal_axis[l]
corp_clean[l] = corp_clean[l] / ||corp_clean[l]||
```

Scaling in Phase 2 uses `corp_clean[l]` — the component of corporate-voice
that is NOT the refusal direction. Safety infrastructure is geometrically
protected.

### 4.5 Per-Direction SVD Scoring

For each sacred layer, for each projection matrix `W` of shape `[out, in]`:

1. SVD-decompose: `U, S, Vh = torch.linalg.svd(W, full_matrices=False)`.
2. For each singular direction index `k`:
   - `in_component = Vh[k]` (the right singular vector — the input-space direction this singular mode reads from)
   - `out_component = U[:, k]` (the left singular vector — what this mode writes to residual stream)
3. Compute three scores per direction:
   - **Classifier-axis projection:** `corporate_magnitude(out_component)` using `corp_clean[l]`. Signed scalar on the corporate axis.
   - **Wanda activation weight:** `|S[k]| * ||proj_acts[l][proj_name] @ Vh[k].T||_2` — how much input-activation energy actually flows through this direction on average.
   - **Dark-path variance:** variance of `authentic_acts[l] @ out_component` — does this direction carry authentic-voice signal?

4. Classification (per direction, thresholds configurable, defaults below):
   - **Bouncer:** `classifier_score > +0.30` (positive, corporate side) AND `wanda_score` in top 50% of the projection's directions AND `dark_variance` in bottom 50%.
   - **Authentic carrier:** `classifier_score < -0.15` AND `dark_variance` in top 50%. **Never touched.**
   - **Neutral/task:** `|classifier_score| < 0.15` AND `wanda_score` > 0. Kept trainable.
   - **Dead:** `wanda_score` in bottom 25% AND `dark_variance` in bottom 25%. Skip (no hook, no scale).

   `classifier_score` is normalized such that the corporate-max-corpus mean
   projection = +1.0 and the authentic-max-corpus mean projection = -1.0.
   The thresholds `+0.30` / `-0.15` express "closer to corporate than to
   one-third of the way from neutral" and "any measurable authentic lean."
   Loosened or tightened per-run via the probe config.

### 4.6 Cross-Layer Propagation Trace

For each bouncer direction identified at layer `l`:
- Project residual-stream activations from the corporate-max corpus at
  layers `[l-2, l-1, l, l+1, l+2]` onto this direction.
- If the signal at layer `l` is a linear extrapolation of `l-1`'s projection
  (correlation > 0.9), it's **inherited**. Mark with `origin_layer = l-1`.
- If layer `l`'s projection is uncorrelated with `l-1`'s, it's **born here**.
  Mark with `origin_layer = l`.

The applicator suppresses only at `origin_layer`, not at downstream
carriers. Surgical, cheap, minimizes collateral.

### 4.7 Magnitude Targets

For each bouncer direction (at its origin layer):
- `current_pos` = classifier-axis projection of `out_component` (from 4.5).
- `target_pos` = `alpha * current_pos` where `alpha = 0.15` by default — attenuates the direction to 15% of its current corporate-axis position, i.e. retains 15% / suppresses 85%. Configurable per-direction.
- **Solve** for the scaling factor `s` such that scaling `S[k] *= s` results in `corporate_magnitude(s * out_component) == target_pos`.
  - Because `corporate_magnitude` is linear in `out_component`, `s = target_pos / current_pos`.
  - Guard: clip `s` to `[0.0, 1.0]` so we never amplify, and never pass through zero into negative space.

Each bouncer direction gets its own `s`. Stored in the atlas.

### 4.8 BrainAtlas Output Schema

```python
@dataclass
class LayerAtlas:
    layer_idx: int
    corporate_axis: torch.Tensor          # [hidden_dim]
    corporate_axis_clean: torch.Tensor    # orthogonalized vs refusal, [hidden_dim]
    refusal_axis: torch.Tensor            # [hidden_dim]
    angle_degrees: float
    neutral_midpoint_projection: float    # bias on the classifier axis
    classifier_coef: torch.Tensor         # raw logistic regression coef, for reproducibility
    per_projection: Dict[str, "ProjectionAtlas"]
    # key: "q_proj" / "k_proj" / ... / "down_proj"
    activation_histogram: Dict[str, torch.Tensor]   # bucket counts on corporate axis, per class

@dataclass
class ProjectionAtlas:
    proj_name: str
    S: torch.Tensor                        # original singular values [rank]
    bouncer_sv_indices: torch.Tensor       # [n_bouncers]
    per_direction_classifier_score: torch.Tensor  # [rank]
    per_direction_wanda_score: torch.Tensor        # [rank]
    per_direction_dark_variance: torch.Tensor      # [rank]
    per_direction_target_scale: torch.Tensor       # [rank], 1.0 for non-bouncers
    origin_layer: Dict[int, int]           # {sv_index: origin_layer_idx}

@dataclass
class BrainAtlas:
    model_name: str
    num_layers: int
    hidden_size: int
    sacred_layers: List[int]
    layers: Dict[int, LayerAtlas]
    probe_config: Dict[str, Any]           # corpus paths, random seeds, classifier params
    built_at: str                          # ISO timestamp

    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "BrainAtlas": ...
```

Serialized as a single `.pt` file. ~10–50 MB for 2B, proportional for 4B.
Self-contained and shippable.

---

## 5. Phase 1 — Aletheia Layer Selection

Independent from Phase 0. Can be run alone on any model.

```python
def run_aletheia(
    model: nn.Module,
    task_batch: List[Dict],
    num_chunks: int = 5,
    chunk_size: int = 8,
    top_k_percent: float = 0.50,
) -> Tuple[List[int], Dict[int, float]]:
    """Forward + backward on task batch; return layers ranked by grad L2 norm."""
```

Implementation:
- Forward-pass the task batch through the model, compute task loss, backward.
- Hook each `lucky_pick_scheduler.deep_chaos.resolve_transformer_layers(model)`
  layer's `self_attn.o_proj.weight.grad` and `mlp.down_proj.weight.grad`.
- Sum L2 norm per layer.
- Return top `top_k_percent` layers.

Default top-50%. Tunable. These are the **sacred_layers** handed to both
the applicator AND `DeepChaosScheduler.sacred_layers`, so chaos doesn't
touch them and sub-zero does.

---

## 6. Phase 2 — Applicator

### 6.1 Public API

```python
def apply_sub_zero(
    model: nn.Module,
    atlas: BrainAtlas,
    sacred_layers: List[int],
) -> SubZeroHandle:
    """
    Mutate sacred-layer projection weights in-place by scaling bouncer
    singular directions toward neutral-midpoint targets. Register grad hooks
    that zero bouncer-dim grads during backward.

    Returns a handle with .remove() and .restore() methods.
    """
```

### 6.2 Per-Projection Mutation

For each sacred layer, for each projection where the atlas contains bouncer
directions (respecting Gemma-4 constraints — see 6.4):

1. Reconstruct the original `W` for verification.
2. `U, S, Vh = svd(W)`.
3. `S_new = S.clone() * atlas.layers[l].per_projection[p].per_direction_target_scale` — per-direction scalar (element-wise), bouncer directions scaled to their solved `s ∈ [0,1]`, non-bouncers multiplied by 1.0 (unchanged).
4. `W_new = U @ diag(S_new) @ Vh`.
5. Write `W_new` into `proj.weight.data` in-place.
6. Register `SVDGradMask` or `DimensionGradMask` on `proj.weight`.

Originals cached in the `SubZeroHandle` for `.restore()`.

### 6.3 Gradient Hooks

Two mask types, choose per projection:

**`SVDGradMask`** (preferred on dense projections):
```python
def hook(grad):
    # Project grad into the SVD basis, zero bouncer singular-direction rows,
    # project back. Prevents the optimizer from re-populating bouncer dirs.
    grad_svd = U_cached.T @ grad @ Vh_cached.T
    grad_svd[bouncer_sv_indices, :] = 0.0
    grad_svd[:, bouncer_sv_indices] = 0.0
    return U_cached @ grad_svd @ Vh_cached
```

**`DimensionGradMask`** (fallback for projections where SVD is numerically
unstable across steps):
```python
def hook(grad):
    masked = grad.clone()
    masked[:, bouncer_col_indices] = 0.0
    return masked
```

Default is `SVDGradMask`. Fall back to `DimensionGradMask` if SVD
reconstruction drifts by more than 1e-4 (checked once at install).

### 6.4 Gemma-4 Architectural Guards

Reuse `lucky_pick_scheduler.deep_chaos` primitives:

```python
from lucky_pick_scheduler.deep_chaos import (
    resolve_transformer_layers,
    LayerBindings,
    _detect_kv_shared,
    _has_post_proj_norm,
)
```

Per sacred layer:
- If `binding.kv_shared` is True, **skip `k_proj` and `v_proj`** entirely
  (they don't exist on this layer; mutating anything would corrupt KV state
  shared by downstream layers).
- Sub-Zero operates on weights, not output activations, so
  `_has_post_proj_norm` is not a blocker here — scaling weights before the
  norm is safe because RMSNorm still sees non-zero inputs (we attenuated,
  not zeroed). This is different from DeepChaos's output-hook approach
  which IS blocked by post-proj norms.
- Still, on layers with post-proj norms, log a warning and verify that
  `||W_new @ x|| > 1e-6` on a smoke-test batch before committing the
  mutation — catch any pathological attenuation early.

### 6.5 Failure Modes & Error Handling

| Failure | Detection | Response |
|---|---|---|
| Classifier fails to separate classes at a layer (accuracy < 60%) | During Phase 0, log per-layer `classifier_accuracy` | Mark layer as "unmapped"; exclude from sacred_layers automatically even if Aletheia ranked it high |
| No bouncer directions found in a projection | `bouncer_sv_indices.numel() == 0` | Log, skip that projection, continue |
| SVD reconstruction drift > 1e-4 | Compare `W` to `U @ diag(S) @ Vh` at install | Fall back to `DimensionGradMask` for that projection |
| Post-apply forward produces NaN on smoke batch | Run smoke forward in `apply_sub_zero` before returning | Abort, call `.restore()`, raise with offending layer |
| Training loss explodes > 2x in first 100 steps | W&B metric tripwire callback | Halt training, call `.restore()`, raise |

---

## 7. Phase 3 — Training Integration

The existing Bella chaos trainer (`bellas/v3/bella-v7-chaos-training.ipynb`
style) is unchanged except for four additions at the top of the notebook:

```python
from sub_zero import build_atlas, apply_sub_zero
from sub_zero.aletheia import run_aletheia
from sub_zero.instrumentation import SubZeroWandbLogger

# Phase 0: build atlas (can be skipped if cached)
atlas = build_atlas(model, tokenizer, corpora_dir="corpora/", cache="atlas.pt")

# Phase 1: pick sacred layers
sacred_layers, _ = run_aletheia(model, task_batch)

# Phase 2: apply
sz_handle = apply_sub_zero(model, atlas, sacred_layers)

# DeepChaosScheduler (existing) — see 7.1 for victim_range constraint
chaos = DeepChaosScheduler.from_model(
    model,
    sacred_layers=sacred_layers,
    victim_range=_compute_victim_range_excluding(sacred_layers, model),
)

# Instrumentation
sz_logger = SubZeroWandbLogger(atlas, sz_handle, chaos)
```

During the training loop, inside the existing per-step callback:
```python
sz_logger.log_step(global_step, model)
```

That's it. No trainer rewrite. No Trainer subclassing. Drop-in.

### 7.1 DeepChaos Interaction Constraint

`DeepChaosScheduler` currently treats `sacred_layers` as "always-active
within the victim range." Sacred layers in the victim range still get
hooked and subsampled (mode forced non-dead, but per-step attn/mlp masks
still apply via the output hooks). That would conflict with Sub-Zero's
base-weight surgery on those same projections — DeepChaos would be zeroing
output channels of weights Sub-Zero just scaled.

Two options, pick at implementation:

**Option A (no DeepChaos change, preferred for Phase B):** choose
`victim_range` to exclude all sacred layers. Works cleanly when sacred
layers are at the ends (e.g. Aletheia picks `[0, 1, 34, 35]` on a 36-layer
model → `victim_range=(2, 34)`). The helper
`_compute_victim_range_excluding()` picks the largest contiguous non-sacred
span and raises with a pointer to Option B if no such span contains enough
layers.

**Option B (small DeepChaos change, preferred for Phase C):** add a
`victim_list` override to `DeepChaosConfig` that accepts an explicit list
instead of a range. One-line change to `_resolve_victim_range` — return
`[i for i in victim_list if i not in self.sacred]`. Becomes the standard
path when Sub-Zero folds into `lucky_pick_scheduler`.

---

## 8. Instrumentation / W&B

Auto-logged every step:

**Phase 2 state:**
- `sub_zero/bouncer_pct/layer_{l}` — fraction of singular directions classified as bouncers per sacred layer
- `sub_zero/authentic_pct/layer_{l}`
- `sub_zero/dead_pct/layer_{l}`
- `sub_zero/corp_refusal_angle_deg/layer_{l}`
- `sub_zero/classifier_accuracy/layer_{l}` (from Phase 0, static)

**Phase 3 dynamics:**
- `sub_zero/corp_axis_alignment/layer_{l}` — cosine of residual-stream mean (computed on a small eval batch every N steps) to `atlas.layers[l].corporate_axis_clean`. **Primary debug signal.** Trending toward zero = suppression is taking.
- `sub_zero/refusal_axis_alignment/layer_{l}` — same for refusal direction. Should stay flat.
- `sub_zero/bouncer_S_mean/layer_{l}_{proj}` — mean of post-scale singular values at bouncer indices. Should stay pinned at the target.
- `sub_zero/grad_mask_leak/layer_{l}_{proj}` — L2 norm of grad at bouncer indices after hook fires. Should be near-zero; if not, hook is broken.

**Voice metric:**
- `sub_zero/corporate_phrase_freq` — count of corporate-voice wordlist matches in N generated responses per checkpoint. Uses the same wordlist that seeded the probe. Single scalar, great demo chart.

**Tripwire:**
- `sub_zero/capability_tripwire` — score on a tiny benchmark (HellaSwag-tiny or equivalent) per checkpoint. Must stay above a configurable floor or training halts.

---

## 9. Testing Strategy

### 9.1 Unit

- `test_hooks_toy_mlp.py` — build a 3-layer MLP, register `SVDGradMask`
  with known bouncer indices, run one forward+backward, assert gradient at
  those indices is zero.
- `test_svd_roundtrip.py` — for a set of random matrices, assert
  `||W - U @ diag(S) @ Vh|| < 1e-5`.
- `test_classifier.py` — synthetic 3-cluster data, assert the classifier
  separates them and the extracted axis is consistent with ground truth.

### 9.2 Integration

- `test_applicator_gemma_e2b_smoke.py` — load `gemma-3-2b-it` in fp16, run
  Phase 0 on a 12-example tiny corpus, run Phase 1 on a 4-example task
  batch, run Phase 2, run one forward on a test prompt, assert no NaN and
  output token ids are within vocab.
- `test_restore.py` — after `apply_sub_zero`, call `.restore()`, assert
  all weights match their pre-apply copies byte-for-byte.

### 9.3 Validation (run manually before e4b port)

- Build the atlas on `gemma-3-2b-it`.
- Render `01_visualize_atlas.ipynb`: heatmap of per-layer
  `corp_refusal_angle_deg`, classifier accuracy, bouncer counts.
- Sanity-check that early layers (0–3) have low bouncer counts (RLHF voice
  is not stored there — embedding and induction head territory).
- Sanity-check that middle-to-late layers (ROME territory) have the highest
  bouncer counts.
- If these intuitions are violated, the probe is wrong and we fix before
  applying.

---

## 10. Open Questions (Pre-Implementation)

1. **Corporate-voice wordlist** — Rick sourcing from Gemini. Blocks Phase 0
   corpus finalization.
2. **Training starting point** — fresh `gemma-3-2b-it` OR merge of existing
   Bella LoRA (`bellas/thebestbella/`) first? Affects probe signal strength.
3. **Compute env** — Modal is the likely host (`modal_train.py` exists).
   Need to confirm VRAM for FFT + optimizer state on 2B and 4B.
4. **K-means k on the classifier residual corpus** — starting with a single
   binary classifier; if corporate voice is multi-modal we may need K=3
   concept cones. Revisit after first atlas render.
5. **Final success metric** — Rick deferred to vibe mid-training. Auto-logged
   A+C metrics ensure data is captured when the decision arrives.

---

## 11. Out of Scope

- Mobile / quantized deployment of the trained model (future concern).
- Non-Gemma architectures — Sub-Zero's mechanics are architecture-agnostic,
  but the Gemma-4 quirks in the applicator make the first release
  Gemma-focused. Port list: Llama-3, Qwen-3, Phi.
- Online probing during training (re-running Phase 0 at checkpoints). The
  atlas is built once and treated as a static map.
- Safety evaluation of the resulting model. Rick retains that call.

---

## 12. Success Criteria for This Spec

The spec is done when:
- Rick can read it and know exactly what modules get built, what each one
  does, and how they hand off.
- Another agent (Codex, Gemini, Perplexity) can pick it up from mempalace
  and continue the work without ambiguity.
- The implementation plan (written next, via `writing-plans` skill) can be
  derived from this doc alone.
