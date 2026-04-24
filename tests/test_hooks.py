import torch

from sub_zero.hooks import DimensionGradMask


def test_dimension_grad_mask_zeroes_columns():
    grad = torch.ones(4, 6)
    mask = DimensionGradMask([1, 4])
    out = mask(grad)
    assert torch.all(out[:, 1] == 0)
    assert torch.all(out[:, 4] == 0)
    assert torch.all(out[:, 0] == 1)
