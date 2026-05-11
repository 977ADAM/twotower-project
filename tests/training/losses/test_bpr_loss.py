from __future__ import annotations

import torch

from twotower._src.training.losses import LossInputs, LossResult
from twotower._src.training.losses.bpr_loss import compute_bpr_loss


def test_bpr_loss_is_lower_when_positive_score_is_higher():
    good = LossInputs(
        positive_scores=torch.tensor([2.0]),
        negative_scores=torch.tensor([0.0]),
    )
    bad = LossInputs(
        positive_scores=torch.tensor([0.0]),
        negative_scores=torch.tensor([2.0]),
    )
    assert compute_bpr_loss(good).loss.item() < compute_bpr_loss(bad).loss.item()


def test_bpr_loss_result_is_scalar_tensor():
    inputs = LossInputs(
        positive_scores=torch.tensor([1.0, 2.0]),
        negative_scores=torch.tensor([0.0, 0.5]),
    )
    result = compute_bpr_loss(inputs)
    assert isinstance(result, LossResult)
    assert result.loss.shape == torch.Size([])


def test_bpr_loss_registry_contains_bpr_key():
    from twotower._src.training.losses import LOSS_REGISTRY
    assert "BPR" in LOSS_REGISTRY
    assert callable(LOSS_REGISTRY["BPR"])
