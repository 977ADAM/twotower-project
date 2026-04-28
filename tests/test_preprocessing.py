from __future__ import annotations

import pandas as pd
import pytest

from twotower._src.backend.config import TwoTowerConfig
from twotower._src.preprocessing import (
    build_evaluation_reference_data,
    build_id_mappings,
    build_labeled_interactions,
    filter_and_sample_interactions,
    prepare_retrieval_pairs,
)


# ── build_labeled_interactions ────────────────────────────────────────────────

def test_build_labeled_interactions_combines_x_and_y():
    X = pd.DataFrame({"user_id": [1, 2], "banner_id": [10, 20]})
    y = [1.0, 0.0]
    df = build_labeled_interactions(X, y, split_name="train")
    assert list(df.columns) == ["user_id", "banner_id", "label"]
    assert df["label"].tolist() == [1.0, 0.0]


def test_build_labeled_interactions_raises_for_missing_columns():
    X = pd.DataFrame({"user_id": [1]})
    with pytest.raises(ValueError, match="banner_id"):
        build_labeled_interactions(X, [1.0], split_name="train")


def test_build_labeled_interactions_raises_for_length_mismatch():
    X = pd.DataFrame({"user_id": [1, 2], "banner_id": [10, 20]})
    with pytest.raises(ValueError, match="same length"):
        build_labeled_interactions(X, [1.0], split_name="train")


def test_build_labeled_interactions_raises_for_non_dataframe():
    with pytest.raises(TypeError):
        build_labeled_interactions([[1, 10]], [1.0], split_name="train")


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
    config = TwoTowerConfig()
    result = filter_and_sample_interactions(
        interactions_df,
        user_id_to_idx={1: 0, 2: 1},
        item_id_to_idx={10: 0, 20: 1, 30: 2},
        config=config,
    )
    assert set(result["user_id"].tolist()) <= {1, 2}
    assert 99 not in result["user_id"].tolist()


def test_filter_applies_max_samples(interactions_df):
    config = TwoTowerConfig(max_samples=2, seed=0)
    result = filter_and_sample_interactions(
        interactions_df,
        user_id_to_idx={1: 0, 2: 1},
        item_id_to_idx={10: 0, 20: 1, 30: 2},
        config=config,
        apply_sampling=True,
    )
    assert len(result) <= 2


def test_filter_raises_for_missing_columns():
    config = TwoTowerConfig()
    df = pd.DataFrame({"user_id": [1], "banner_id": [10]})
    with pytest.raises(ValueError, match="label"):
        filter_and_sample_interactions(df, user_id_to_idx={1: 0}, item_id_to_idx={10: 0}, config=config)


# ── prepare_retrieval_pairs ───────────────────────────────────────────────────

def test_prepare_retrieval_pairs_returns_only_positives(interactions_df):
    config = TwoTowerConfig()
    result = prepare_retrieval_pairs(
        interactions_df,
        user_id_to_idx={1: 0, 2: 1},
        item_id_to_idx={10: 0, 20: 1, 30: 2},
        config=config,
        apply_sampling=False,
        split_name="train",
    )
    assert (result["label"] == 1.0).all()


def test_prepare_retrieval_pairs_raises_when_no_positives():
    config = TwoTowerConfig()
    df = pd.DataFrame({"user_id": [1], "banner_id": [10], "label": [0.0]})
    with pytest.raises(ValueError, match="no positive interactions"):
        prepare_retrieval_pairs(df, user_id_to_idx={1: 0}, item_id_to_idx={10: 0}, config=config, apply_sampling=False, split_name="train")


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
