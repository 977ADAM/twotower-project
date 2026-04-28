from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from twotower import EarlyStopping, TwoTower


@pytest.fixture
def small_interactions():
    """Minimal train/valid/test DataFrames with 5 users and 10 items."""
    rows = [
        (u, i, 1.0) for u in range(1, 6) for i in range(1, 6)
    ] + [
        (u, i, 0.0) for u in range(1, 6) for i in range(6, 11)
    ]
    df = pd.DataFrame(rows, columns=["user_id", "banner_id", "label"])
    train = df.copy()
    valid = df.copy()
    test = df.copy()
    return train, valid, test


@pytest.fixture
def fitted_model(small_interactions):
    train, valid, _ = small_interactions
    model = TwoTower(epochs=2, batch_size=8, eval_during_training=False, device="cpu", seed=0)
    model.fit(
        X_train=train[["user_id", "banner_id"]],
        y_train=train["label"],
        X_valid=valid[["user_id", "banner_id"]],
        y_valid=valid["label"],
        early_stopping=None,
    )
    return model


# ── fit ───────────────────────────────────────────────────────────────────────

def test_fit_returns_history_with_train_and_valid_loss(fitted_model):
    history = fitted_model.train_history
    assert len(history) == 2
    for record in history:
        assert "train_loss" in record
        assert "valid_loss" in record


def test_fit_populates_id_mappings(fitted_model):
    assert len(fitted_model.idx_to_user_id) == 5
    assert len(fitted_model.idx_to_item_id) == 10


def test_fit_with_early_stopping_can_halt_early(small_interactions):
    train, valid, _ = small_interactions
    # lr=0 → loss never changes → early stopping triggers after patience+1 epochs
    model = TwoTower(epochs=20, batch_size=8, learning_rate=0.0, eval_during_training=False, device="cpu", seed=0)
    history = model.fit(
        X_train=train[["user_id", "banner_id"]],
        y_train=train["label"],
        X_valid=valid[["user_id", "banner_id"]],
        y_valid=valid["label"],
        early_stopping=EarlyStopping(patience=2, min_delta=0.0),
    )
    assert len(history) < 20


# ── predict ───────────────────────────────────────────────────────────────────

def test_predict_returns_top_k_items_per_user(fitted_model):
    # exclude_seen=False so seen items don't reduce the candidate set
    predictions = fitted_model.predict(user_ids=[1, 2], top_k=3, exclude_seen=False)
    assert set(predictions.keys()) == {1, 2}
    for recs in predictions.values():
        assert len(recs) == 3
        assert all("banner_id" in r and "score" in r for r in recs)  # default item_col


def test_predict_excludes_seen_items_by_default(fitted_model):
    seen = fitted_model.get_seen_items_by_user()
    predictions = fitted_model.predict(user_ids=[1], top_k=5, exclude_seen=True)
    predicted_ids = {r["banner_id"] for r in predictions[1]}
    assert predicted_ids.isdisjoint(seen.get(1, set()))


def test_predict_raises_before_fit():
    model = TwoTower()
    with pytest.raises(RuntimeError, match="not fitted"):
        model.predict()


def test_predict_strict_raises_for_unknown_user(fitted_model):
    with pytest.raises(ValueError, match="unknown user_ids"):
        fitted_model.predict(user_ids=[9999], strict=True)


# ── evaluate ──────────────────────────────────────────────────────────────────

def test_evaluate_returns_recall_and_loss_metrics(fitted_model, small_interactions):
    _, _, test = small_interactions
    metrics = fitted_model.evaluate(test)
    assert "recall_at_k" in metrics
    assert "test_loss" in metrics
    assert 0.0 <= metrics["recall_at_k"] <= 1.0


# ── save / load ───────────────────────────────────────────────────────────────

def test_save_and_load_produces_identical_predictions(fitted_model, small_interactions):
    _, _, test = small_interactions
    original_preds = fitted_model.predict(user_ids=[1, 2], top_k=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.pth"
        fitted_model.save_model(path)

        loaded = TwoTower().load_model(path)

    loaded_preds = loaded.predict(user_ids=[1, 2], top_k=5)
    assert original_preds == loaded_preds


def test_load_model_raises_for_missing_file():
    with pytest.raises(FileNotFoundError):
        TwoTower().load_model("nonexistent.pth")
