from __future__ import annotations

from twotower._src.training.progress import EpochSummary, _build_epoch_line


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


def test_no_prev_metrics_shows_no_arrows():
    summary = EpochSummary(metrics={"train_loss": 0.5, "valid_loss": 0.4})
    line = _build_epoch_line(3, 5, summary, prev_metrics={})
    assert "↑" not in line
    assert "↓" not in line
    assert "→" not in line
    assert "0.5000" in line
    assert "0.4000" in line


def test_loss_decrease_shows_green_down_arrow():
    summary = EpochSummary(metrics={"train_loss": 0.4})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"train_loss": 0.5})
    assert "↓" in line
    assert "green" in line


def test_loss_increase_shows_red_up_arrow():
    summary = EpochSummary(metrics={"train_loss": 0.6})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"train_loss": 0.5})
    assert "↑" in line
    assert "red" in line


def test_recall_increase_shows_green_up_arrow():
    summary = EpochSummary(metrics={"recall_at_50": 0.6})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"recall_at_50": 0.5})
    assert "↑" in line
    assert "green" in line


def test_recall_decrease_shows_red_down_arrow():
    summary = EpochSummary(metrics={"recall_at_50": 0.4})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"recall_at_50": 0.5})
    assert "↓" in line
    assert "red" in line


def test_tiny_change_shows_dim_neutral_arrow():
    summary = EpochSummary(metrics={"train_loss": 0.50001})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"train_loss": 0.5})
    assert "→" in line
    assert "dim" in line


def test_is_best_shows_star_marker():
    summary = EpochSummary(metrics={"train_loss": 0.4}, is_best=True)
    line = _build_epoch_line(2, 5, summary, prev_metrics={})
    assert "★" in line
    assert "best" in line


def test_patience_counter_shown_when_positive():
    summary = EpochSummary(metrics={"train_loss": 0.5}, patience_used=2, patience_total=5)
    line = _build_epoch_line(2, 5, summary, prev_metrics={})
    assert "no improvement" in line
    assert "2/5" in line


def test_patience_counter_hidden_when_zero():
    summary = EpochSummary(metrics={"train_loss": 0.5}, patience_used=0, patience_total=5)
    line = _build_epoch_line(2, 5, summary, prev_metrics={})
    assert "no improvement" not in line


def test_patience_counter_hidden_when_no_early_stopping():
    summary = EpochSummary(metrics={"train_loss": 0.5}, patience_used=3, patience_total=None)
    line = _build_epoch_line(2, 5, summary, prev_metrics={})
    assert "no improvement" not in line
