from __future__ import annotations

from sub_zero.aletheia import run_aletheia

from conftest import build_task_batches


def test_run_aletheia_returns_scores_for_all_layers(toy_model, toy_tokenizer):
    task_batches = build_task_batches(
        toy_tokenizer,
        prompts=[
            "debug this issue",
            "reduce latency in training",
            "write clear test coverage",
        ],
    )
    sacred, scores = run_aletheia(
        toy_model,
        task_batches=task_batches,
        num_probe_batches=1,
        top_k_percent=0.5,
    )

    n_layers = toy_model.config.num_hidden_layers
    assert len(scores) == n_layers
    assert len(sacred) == max(1, round(n_layers * 0.5))
    assert all(0 <= idx < n_layers for idx in sacred)
