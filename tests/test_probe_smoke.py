from __future__ import annotations

from pathlib import Path

from sub_zero.probe import ProbeConfig, build_atlas

from conftest import build_task_batches


def test_probe_builds_and_serializes(toy_model, toy_tokenizer, tmp_path):
    cfg = ProbeConfig(
        corpora_dir="/Users/chiggy/sub-zero/corpora",
        max_prompts_per_class=8,
        max_length=64,
        classifier_accuracy_floor=0.0,
        sacred_top_k_percent=0.5,
    )
    task_batches = build_task_batches(
        toy_tokenizer,
        prompts=[
            "fix the scheduler bottleneck quickly",
            "explain why this optimization works",
            "apply the patch and rerun tests",
        ],
    )
    cache_path = tmp_path / "atlas.pt"
    atlas = build_atlas(
        toy_model,
        toy_tokenizer,
        config=cfg,
        task_batches=task_batches,
        cache_path=str(cache_path),
    )

    assert atlas.num_layers == toy_model.config.num_hidden_layers
    assert len(atlas.layers) > 0
    assert cache_path.exists()

    # sanity on one layer payload
    layer0 = atlas.layers[min(atlas.layers.keys())]
    assert hasattr(layer0, "classifier_accuracy")
    assert layer0.corporate_axis.numel() == toy_model.config.hidden_size
