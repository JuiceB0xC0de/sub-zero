import torch

from sub_zero.atlas import BrainAtlas, LayerAtlas, ProjectionAtlas


def test_atlas_roundtrip(tmp_path):
    pa = ProjectionAtlas(
        proj_name="q_proj",
        S=torch.ones(4),
        bouncer_sv_indices=torch.tensor([1, 2]),
        per_direction_classifier_score=torch.randn(4),
        per_direction_wanda_score=torch.randn(4).abs(),
        per_direction_dark_variance=torch.randn(4).abs(),
        per_direction_target_scale=torch.ones(4),
        origin_layer={1: 3},
    )
    la = LayerAtlas(
        layer_idx=3,
        corporate_axis=torch.randn(8),
        corporate_axis_clean=torch.randn(8),
        refusal_axis=torch.randn(8),
        angle_degrees=75.0,
        neutral_midpoint_projection=0.1,
        classifier_coef=torch.randn(8),
        per_projection={"q_proj": pa},
        activation_histogram={"corporate": torch.tensor([1.0, 2.0, 3.0])},
        classifier_accuracy=0.91,
    )
    atlas = BrainAtlas(
        model_name="test/model",
        num_layers=4,
        hidden_size=8,
        sacred_layers=[2, 3],
        layers={3: la},
        probe_config={"x": 1},
    )

    path = tmp_path / "atlas.pt"
    atlas.save(str(path))
    loaded = BrainAtlas.load(str(path))
    assert loaded.model_name == "test/model"
    assert loaded.layers[3].per_projection["q_proj"].bouncer_sv_indices.tolist() == [1, 2]
