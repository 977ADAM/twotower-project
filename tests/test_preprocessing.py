from __future__ import annotations

import pandas as pd
import pytest

from twotower._src.config import _Config
from twotower._src.data.preprocessing import (
    build_evaluation_reference_data,
    build_id_mappings,
    filter_and_sample_interactions,
    normalize_fit_interactions,
    prepare_retrieval_pairs,
)

# ── normalize_fit_interactions ────────────────────────────────────────────────

def test_normalize_fit_interactions_returns_correct_columns():
    df = pd.DataFrame({"user_id": [1, 2], "banner_id": [10, 20], "label": [1.0, 0.0]})
    result = normalize_fit_interactions(df, split_name="train")
    assert list(result.columns) == ["user_id", "banner_id", "label"]
    assert result["label"].tolist() == [1.0, 0.0]


def test_normalize_fit_interactions_raises_for_missing_label():
    df = pd.DataFrame({"user_id": [1], "banner_id": [10]})
    with pytest.raises(ValueError, match="label"):
        normalize_fit_interactions(df, split_name="train")


def test_normalize_fit_interactions_raises_for_missing_item_col():
    df = pd.DataFrame({"user_id": [1], "label": [1.0]})
    with pytest.raises(ValueError, match="banner_id"):
        normalize_fit_interactions(df, split_name="train")


def test_normalize_fit_interactions_raises_for_non_dataframe():
    with pytest.raises(TypeError):
        normalize_fit_interactions([[1, 10, 1.0]], split_name="train")


# ── build_id_mappings ─────────────────────────────────────────────────────────

def test_build_id_mappings_creates_correct_bidirectional_mappings():
    df = pd.DataFrame({"user_id": [3, 1, 2], "banner_id": [30, 10, 30], "label": [1.0, 1.0, 0.0]})
    mappings = build_id_mappings(df)
    assert mappings.idx_to_user_id == [1, 2, 3]
    assert mappings.user_id_to_idx == {1: 0, 2: 1, 3: 2}
    assert mappings.idx_to_item_id == [10, 30]
    assert mappings.item_id_to_idx == {10: 0, 30: 1}


# ── filter_and_sample_interactions ───────────────────────────────────────────

@pytest.fixture
def interactions_df():
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 99],
        "banner_id": [10, 20, 10, 30, 10],
        "label": [1.0, 0.0, 1.0, 1.0, 1.0],
    })


def test_filter_removes_unknown_users_and_items(interactions_df):
    config = _Config()
    result = filter_and_sample_interactions(
        interactions_df,
        user_id_to_idx={1: 0, 2: 1},
        item_id_to_idx={10: 0, 20: 1, 30: 2},
        config=config,
    )
    assert set(result["user_id"].tolist()) <= {1, 2}
    assert 99 not in result["user_id"].tolist()


def test_filter_raises_for_missing_columns():
    config = _Config()
    df = pd.DataFrame({"user_id": [1], "banner_id": [10]})
    with pytest.raises(ValueError, match="label"):
        filter_and_sample_interactions(df, user_id_to_idx={1: 0}, item_id_to_idx={10: 0}, config=config)


# ── prepare_retrieval_pairs ───────────────────────────────────────────────────

def test_prepare_retrieval_pairs_returns_only_positives(interactions_df):
    config = _Config()
    result = prepare_retrieval_pairs(
        interactions_df,
        user_id_to_idx={1: 0, 2: 1},
        item_id_to_idx={10: 0, 20: 1, 30: 2},
        config=config,
        split_name="train",
    )
    assert (result["label"] == 1.0).all()


def test_prepare_retrieval_pairs_raises_when_no_positives():
    config = _Config()
    df = pd.DataFrame({"user_id": [1], "banner_id": [10], "label": [0.0]})
    with pytest.raises(ValueError, match="no positive interactions"):
        prepare_retrieval_pairs(
            df, user_id_to_idx={1: 0}, item_id_to_idx={10: 0}, config=config, split_name="train"
        )


# ── build_evaluation_reference_data ──────────────────────────────────────────

def test_build_evaluation_reference_data_returns_seen_items_and_popularity():
    train_df = pd.DataFrame({
        "user_id": [1, 1, 2],
        "banner_id": [10, 20, 10],
        "label": [1.0, 1.0, 1.0],
    })
    seen, popularity = build_evaluation_reference_data(train_df, None)
    assert seen[1] == {10, 20}
    assert seen[2] == {10}
    assert popularity[0] == 10  # item 10 appears twice → most popular


def test_build_evaluation_reference_data_merges_train_and_valid():
    train_df = pd.DataFrame({"user_id": [1], "banner_id": [10], "label": [1.0]})
    valid_df = pd.DataFrame({"user_id": [1], "banner_id": [20], "label": [1.0]})
    seen, _ = build_evaluation_reference_data(train_df, valid_df)
    assert seen[1] == {10, 20}


def test_build_evaluation_reference_data_handles_none_inputs():
    seen, popularity = build_evaluation_reference_data(None, None)
    assert seen == {}
    assert popularity == []
