from __future__ import annotations

import pandas as pd
import pytest

from twotower._src.data import normalize_interactions, split_interactions


def _make_interactions(dates: list[str], clicks: list[int] | None = None) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame({
        "event_date": dates,
        "user_id": list(range(1, n + 1)),
        "banner_id": list(range(101, n + 101)),
        "clicks": clicks if clicks is not None else [1] * n,
    })


# ── normalize_interactions ────────────────────────────────────────────────────

def test_normalize_interactions_derives_label_from_clicks():
    df = _make_interactions(["2024-01-01", "2024-01-02"], clicks=[0, 2])
    result = normalize_interactions(df)
    assert result["label"].tolist() == [0.0, 1.0]


def test_normalize_interactions_sorts_by_date():
    df = _make_interactions(["2024-01-03", "2024-01-01", "2024-01-02"])
    result = normalize_interactions(df)
    dates = pd.to_datetime(result["event_date"]).tolist()
    assert dates == sorted(dates)


def test_normalize_interactions_raises_for_missing_columns():
    df = pd.DataFrame({"event_date": ["2024-01-01"], "user_id": [1], "banner_id": [10]})
    with pytest.raises(ValueError, match="clicks"):
        normalize_interactions(df)


# ── split_interactions ────────────────────────────────────────────────────────

@pytest.fixture
def multi_date_df():
    dates = (
        ["2024-01-01"] * 70
        + ["2024-01-02"] * 20
        + ["2024-01-03"] * 10
    )
    return _make_interactions(dates)


def test_split_interactions_produces_non_empty_splits(multi_date_df):
    train, valid, test = split_interactions(multi_date_df, validation_ratio=0.2, test_ratio=0.1)
    assert len(train) > 0
    assert len(valid) > 0
    assert len(test) > 0


def test_split_interactions_splits_are_chronological(multi_date_df):
    train, valid, test = split_interactions(multi_date_df, validation_ratio=0.2, test_ratio=0.1)
    train_max = pd.to_datetime(train["event_date"]).max()
    valid_min = pd.to_datetime(valid["event_date"]).min()
    valid_max = pd.to_datetime(valid["event_date"]).max()
    test_min = pd.to_datetime(test["event_date"]).min()
    assert train_max < valid_min
    assert valid_max < test_min


def test_split_interactions_no_overlap(multi_date_df):
    train, valid, test = split_interactions(multi_date_df)
    all_indices = set(train.index) | set(valid.index) | set(test.index)
    assert len(all_indices) == len(train) + len(valid) + len(test)


def test_split_interactions_raises_for_empty_df():
    df = _make_interactions([])
    with pytest.raises(ValueError, match="empty"):
        split_interactions(df)


def test_split_interactions_raises_when_ratios_sum_to_one():
    df = _make_interactions(["2024-01-01", "2024-01-02", "2024-01-03"])
    with pytest.raises(ValueError, match="less than 1"):
        split_interactions(df, validation_ratio=0.5, test_ratio=0.5)


def test_split_interactions_raises_for_too_few_dates():
    df = _make_interactions(["2024-01-01", "2024-01-01"])
    with pytest.raises(ValueError, match="3 unique event dates"):
        split_interactions(df)
