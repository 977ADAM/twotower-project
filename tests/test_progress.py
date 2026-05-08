from __future__ import annotations

from twotower._src.training.progress import EpochSummary


def test_epoch_summary_defaults():
    s = EpochSummary(metrics={"train_loss": 0.5})
    assert s.is_best is False
    assert s.patience_used == 0
    assert s.patience_total is None


def test_epoch_summary_all_fields():
    s = EpochSummary(metrics={"train_loss": 0.5}, is_best=True, patience_used=2, patience_total=5)
    assert s.is_best is True
    assert s.patience_used == 2
    assert s.patience_total == 5
