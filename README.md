# sub-zero

> ⚠️ **Work in progress — incomplete.** This repository is an active research prototype, not a finished tool. The probe pipeline runs end-to-end, but the applicator/training-integration boundary still has known gaps, the gate thresholds are tuned to a single Gemma-4-E2B-it run, and no public stability or correctness guarantees are made. Expect breaking changes, partial documentation, and rough edges throughout. If you are reading this looking for a stable interpretability tool, this isn't one yet.

Sub-Zero implementation package for hidden-dimension selective freezing.

## Modules

- `sub_zero.probe`: builds `BrainAtlas` from corpora prompts.
- `sub_zero.aletheia`: gradient-guided sacred layer selection.
- `sub_zero.applicator`: applies SVD scaling + gradient masking.
- `sub_zero.instrumentation`: W&B-friendly metric payload helpers.

## Quick start

```python
from sub_zero import ProbeConfig, build_atlas, run_aletheia, apply_sub_zero

cfg = ProbeConfig(corpora_dir="./corpora")
atlas = build_atlas(model, tokenizer, cfg, task_batches=task_batches, cache_path="./atlas.pt")
handle = apply_sub_zero(model, atlas, sacred_layers=atlas.sacred_layers)
```

Call `handle.restore(model)` to restore original weights.

## Tests

```bash
PYTHONPATH=. python -m pytest -q tests
```

Current suite covers:
- atlas roundtrip serialization
- gradient mask behavior
- Aletheia layer scoring
- probe smoke run
- applicator install/restore
- end-to-end one-step integration

## CLI

```bash
sub-zero --model google/gemma-3-2b-it --corpora-dir ./corpora --out ./atlas.pt
```
