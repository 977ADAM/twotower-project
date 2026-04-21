from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeAlias

import pandas as pd

from twotower.src.backend.config import TwoTowerConfig
from twotower.data import normalize_interactions

TargetLike: TypeAlias = pd.Series | Sequence[float]


@dataclass(slots=True)
class IdMappings:
    user_id_to_idx: dict[int, int]
    item_id_to_idx: dict[int, int]
    idx_to_user_id: list[int]
    idx_to_item_id: list[int]


def build_labeled_interactions(
    X: pd.DataFrame,
    y: TargetLike,
    split_name: str,
) -> pd.DataFrame:
    """Validate X and y, combine into a labeled interactions DataFrame."""
    if not isinstance(X, pd.DataFrame):
        raise TypeError(
            f"{split_name} features must be a pandas DataFrame with "
            "'user_id' and 'banner_id' columns."
        )
    required_columns = {"user_id", "banner_id"}
    missing_columns = required_columns.difference(X.columns)
    if missing_columns:
        raise ValueError(
            f"{split_name} features are missing required columns: {sorted(missing_columns)}"
        )

    y_series = y if isinstance(y, pd.Series) else pd.Series(y, name="label")
    if len(X) != len(y_series):
        raise ValueError(
            f"{split_name} features and labels must have the same length: "
            f"{len(X)} != {len(y_series)}"
        )

    prepared_df = X.loc[:, ["user_id", "banner_id"]].copy()
    prepared_df["label"] = y_series.to_numpy()
    return prepared_df.reset_index(drop=True)


def build_id_mappings(train_df: pd.DataFrame) -> IdMappings:
    """Build bidirectional user/item ID ↔ index mappings from training data."""
    idx_to_user_id = train_df["user_id"].astype(int).drop_duplicates().sort_values().tolist()
    idx_to_item_id = train_df["banner_id"].astype(int).drop_duplicates().sort_values().tolist()
    return IdMappings(
        user_id_to_idx={user_id: idx for idx, user_id in enumerate(idx_to_user_id)},
        item_id_to_idx={item_id: idx for idx, item_id in enumerate(idx_to_item_id)},
        idx_to_user_id=idx_to_user_id,
        idx_to_item_id=idx_to_item_id,
    )


def filter_and_sample_interactions(
    interactions_df: pd.DataFrame,
    *,
    user_id_to_idx: dict[int, int],
    item_id_to_idx: dict[int, int],
    config: TwoTowerConfig,
    apply_sampling: bool = False,
    sort_by_event_date: bool = False,
) -> pd.DataFrame:
    """Filter to known user/item IDs, optionally balance-sample, and sort by date."""
    required_columns = {"user_id", "banner_id", "label"}
    missing_columns = required_columns.difference(interactions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Prepared interactions dataframe is missing columns: {sorted(missing_columns)}"
        )

    selected_columns = ["user_id", "banner_id", "label"]
    if "event_date" in interactions_df.columns:
        selected_columns.append("event_date")

    interactions = interactions_df.loc[:, selected_columns].copy()
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["banner_id"] = interactions["banner_id"].astype(int)
    interactions["label"] = interactions["label"].astype("float32")
    interactions = interactions[
        interactions["user_id"].isin(user_id_to_idx)
        & interactions["banner_id"].isin(item_id_to_idx)
    ]

    if apply_sampling and config.max_samples and len(interactions) > config.max_samples:
        positives = interactions[interactions["label"] == 1.0]
        negatives = interactions[interactions["label"] == 0.0]
        positive_target = min(len(positives), config.max_samples // 2)
        negative_target = min(len(negatives), config.max_samples - positive_target)

        sampled_frames = []
        if positive_target:
            sampled_frames.append(
                positives.sample(n=positive_target, random_state=config.seed, replace=False)
            )
        if negative_target:
            sampled_frames.append(
                negatives.sample(n=negative_target, random_state=config.seed, replace=False)
            )
        interactions = pd.concat(sampled_frames, ignore_index=True)

    if sort_by_event_date and "event_date" in interactions.columns:
        interactions = interactions.sort_values("event_date")

    return interactions.reset_index(drop=True)


def prepare_retrieval_pairs(
    interactions_df: pd.DataFrame,
    *,
    user_id_to_idx: dict[int, int],
    item_id_to_idx: dict[int, int],
    config: TwoTowerConfig,
    apply_sampling: bool,
    split_name: str,
) -> pd.DataFrame:
    """Filter to positive interactions only; optionally sample to max_samples."""
    filtered_interactions = filter_and_sample_interactions(
        interactions_df,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        config=config,
        apply_sampling=False,
        sort_by_event_date=False,
    )
    positive_interactions = filtered_interactions[filtered_interactions["label"] == 1.0].copy()
    if positive_interactions.empty:
        raise ValueError(f"{split_name} split has no positive interactions for retrieval training.")

    if apply_sampling and config.max_samples and len(positive_interactions) > config.max_samples:
        positive_interactions = positive_interactions.sample(
            n=config.max_samples,
            random_state=config.seed,
            replace=False,
        )

    return positive_interactions.reset_index(drop=True)


def prepare_evaluation_inputs(X_test: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize evaluation DataFrame to (event_date, user_id, banner_id, label) format."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "Evaluation features must be a pandas DataFrame with "
            "'user_id' and 'banner_id' columns."
        )

    required_columns = {"user_id", "banner_id"}
    missing_columns = required_columns.difference(X_test.columns)
    if missing_columns:
        raise ValueError(
            f"Evaluation features are missing required columns: {sorted(missing_columns)}"
        )

    if "label" in X_test.columns:
        evaluation_df = X_test.copy()
        evaluation_df["user_id"] = evaluation_df["user_id"].astype(int)
        evaluation_df["banner_id"] = evaluation_df["banner_id"].astype(int)
        evaluation_df["label"] = evaluation_df["label"].astype("float32")
        if "event_date" not in evaluation_df.columns:
            evaluation_df["event_date"] = pd.Timestamp("1970-01-01")
        else:
            evaluation_df["event_date"] = pd.to_datetime(evaluation_df["event_date"])
        return evaluation_df.loc[:, ["event_date", "user_id", "banner_id", "label"]]

    if "clicks" not in X_test.columns:
        raise ValueError(
            "Evaluation features must include either a 'label' column or a 'clicks' column."
        )

    evaluation_df = X_test.copy()
    if "event_date" not in evaluation_df.columns:
        evaluation_df["event_date"] = pd.Timestamp("1970-01-01")
    return normalize_interactions(evaluation_df)


def normalize_and_filter_interactions(
    interactions_df: pd.DataFrame,
    *,
    user_id_to_idx: dict[int, int],
    item_id_to_idx: dict[int, int],
    config: TwoTowerConfig,
    apply_sampling: bool = False,
) -> pd.DataFrame:
    """Normalize raw or pre-labeled interactions, then filter to known IDs."""
    if "label" in interactions_df.columns:
        prepared = interactions_df.copy()
        if "event_date" not in prepared.columns:
            prepared["event_date"] = pd.Timestamp("1970-01-01")
    else:
        prepared = normalize_interactions(interactions_df)

    return filter_and_sample_interactions(
        prepared,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        config=config,
        apply_sampling=apply_sampling,
        sort_by_event_date=True,
    )


def build_evaluation_reference_data(
    train_df: pd.DataFrame | None,
    valid_df: pd.DataFrame | None,
) -> tuple[dict[int, set[int]], list[int]]:
    """Return seen_items_by_user and popularity-ranked positive item IDs from train/valid data."""
    seen_items_by_user: dict[int, set[int]] = {}
    for dataframe in (train_df, valid_df):
        if dataframe is None or dataframe.empty:
            continue
        grouped = dataframe.groupby("user_id")["banner_id"]
        for user_id, item_ids in grouped:
            seen_items_by_user.setdefault(int(user_id), set()).update(
                int(item_id) for item_id in item_ids.tolist()
            )

    if train_df is None or train_df.empty:
        return seen_items_by_user, []

    train_positive_item_ids_by_popularity = (
        train_df.loc[train_df["label"] == 1.0, "banner_id"]
        .astype(int)
        .value_counts()
        .index
        .tolist()
    )
    return seen_items_by_user, train_positive_item_ids_by_popularity
