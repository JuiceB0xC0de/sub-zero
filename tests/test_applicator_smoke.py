from __future__ import annotations

import torch

from sub_zero.applicator import apply_sub_zero
from sub_zero.atlas import BrainAtlas, LayerAtlas, ProjectionAtlas

from conftest import build_task_batches


def _manual_atlas_for_bouncer(toy_model) -> BrainAtlas:
    layer_idx = 0
    proj_name = "q_proj"
    pmod = toy_model.layers[layer_idx].self_attn.q_proj
    w = pmod.weight.detach().float().cpu()
    u, s, vh = torch.linalg.svd(w, full_matrices=False)

    rank = s.numel()
    bouncer_idx = torch.tensor([0], dtype=torch.long)
    scales = torch.ones(rank)
    scales[0] = 0.15

    patlas = ProjectionAtlas(
        proj_name=proj_name,
        S=s,
        bouncer_sv_indices=bouncer_idx,
        per_direction_classifier_score=torch.zeros(rank),
        per_direction_wanda_score=torch.ones(rank),
        per_direction_dark_variance=torch.ones(rank),
        per_direction_target_scale=scales,
        origin_layer={0: layer_idx},
    )

    hidden = toy_model.config.hidden_size
    latlas = LayerAtlas(
        layer_idx=layer_idx,
        corporate_axis=torch.zeros(hidden),
        corporate_axis_clean=torch.zeros(hidden),
        refusal_axis=torch.zeros(hidden),
        angle_degrees=90.0,
        neutral_midpoint_projection=0.0,
        classifier_coef=torch.zeros(hidden),
        per_projection={proj_name: patlas},
        activation_histogram={},
        classifier_accuracy=1.0,
    )

    return BrainAtlas(
        model_name="toy/model",
        num_layers=toy_model.config.num_hidden_layers,
        hidden_size=hidden,
        sacred_layers=[layer_idx],
        layers={layer_idx: latlas},
        probe_config={},
    )


def test_apply_installs_hook_and_restore(toy_model, toy_tokenizer):
    atlas = _manual_atlas_for_bouncer(toy_model)

    before = toy_model.layers[0].self_attn.q_proj.weight.detach().clone()
    handle = apply_sub_zero(toy_model, atlas)
    assert len(handle.hook_handles) >= 1

    # one train step
    task_batches = build_task_batches(toy_tokenizer, ["test training step", "another sample"])
    batch = task_batches[0]
    out = toy_model(**batch)
    out.loss.backward()

    handle.restore(toy_model)
    after = toy_model.layers[0].self_attn.q_proj.weight.detach().clone()
    assert torch.allclose(before, after, atol=1e-6)
