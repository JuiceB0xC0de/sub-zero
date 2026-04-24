from __future__ import annotations

import torch

from sub_zero.applicator import apply_sub_zero
from sub_zero.probe import ProbeConfig, build_atlas

from conftest import build_task_batches


def test_end_to_end_pipeline_one_step(toy_model, toy_tokenizer, tmp_path):
    cfg = ProbeConfig(
        corpora_dir="/Users/chiggy/sub-zero/corpora",
        max_prompts_per_class=8,
        max_length=64,
        classifier_accuracy_floor=0.0,
        sacred_top_k_percent=0.5,
        num_probe_batches=1,
    )
    task_batches = build_task_batches(
        toy_tokenizer,
        prompts=[
            "optimize this code path",
            "explain tradeoffs clearly",
            "apply patch and verify",
        ],
    )

    atlas = build_atlas(
        toy_model,
        toy_tokenizer,
        config=cfg,
        task_batches=task_batches,
        cache_path=str(tmp_path / "atlas.pt"),
    )

    handle = apply_sub_zero(toy_model, atlas)

    opt = torch.optim.AdamW(toy_model.parameters(), lr=1e-3)
    batch = task_batches[0]
    opt.zero_grad(set_to_none=True)
    out = toy_model(**batch)
    assert out.loss is not None
    assert torch.isfinite(out.loss).item()
    out.loss.backward()
    opt.step()

    # restore original touched projections and ensure no crash
    handle.restore(toy_model)
    handle.remove()
