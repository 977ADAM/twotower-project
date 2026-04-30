from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from twotower._src.config import _Config
from twotower._src.data.split import normalize_interactions


@dataclass(slots=True)
class IdMappings:
    query_id_to_idx: dict[int, int]
    candidate_id_to_idx: dict[int, int]
    idx_to_query_id: list[int]
    idx_to_candidate_id: list[int]


def normalize_fit_interactions(
    df: pd.DataFrame,
    split_name: str,
    query_col: str = "query_id",
    candidate_col: str = "candidate_id",
) -> pd.DataFrame:
    """Validate and normalize a labeled interactions DataFrame for fitting."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{split_name} interactions must be a pandas DataFrame with "
            f"'{query_col}', '{candidate_col}', and 'label' columns."
        )
    required_columns = {query_col, candidate_col, "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"{split_name} interactions are missing required columns: {sorted(missing_columns)}"
        )

    prepared_df = df.loc[:, [query_col, candidate_col, "label"]].copy()
    prepared_df = prepared_df.rename(columns={query_col: "query_id", candidate_col: "candidate_id"})
    return prepared_df.reset_index(drop=True)


def build_id_mappings(train_df: pd.DataFrame) -> IdMappings:
    """Build bidirectional user/item ID ↔ index mappings from training data."""
    idx_to_query_id = train_df["query_id"].astype(int).drop_duplicates().sort_values().tolist()
    idx_to_candidate_id = train_df["candidate_id"].astype(int).drop_duplicates().sort_values().tolist()
    return IdMappings(
        query_id_to_idx={query_id: idx for idx, query_id in enumerate(idx_to_query_id)},
        candidate_id_to_idx={candidate_id: idx for idx, candidate_id in enumerate(idx_to_candidate_id)},
        idx_to_query_id=idx_to_query_id,
        idx_to_candidate_id=idx_to_candidate_id,
    )


def filter_and_sample_interactions(
    interactions_df: pd.DataFrame,
    *,
    query_id_to_idx: dict[int, int],
    candidate_id_to_idx: dict[int, int],
    config: _Config,
    sort_by_event_date: bool = False,
) -> pd.DataFrame:
    """Filter to known user/item IDs and sort by date."""
    required_columns = {"query_id", "candidate_id", "label"}
    missing_columns = required_columns.difference(interactions_df.columns)
    if missing_columns:
        raise ValueError(
            f"Prepared interactions dataframe is missing columns: {sorted(missing_columns)}"
        )

    selected_columns = ["query_id", "candidate_id", "label"]
    if "event_date" in interactions_df.columns:
        selected_columns.append("event_date")

    interactions = interactions_df.loc[:, selected_columns].copy()
    interactions["query_id"] = interactions["query_id"].astype(int)
    interactions["candidate_id"] = interactions["candidate_id"].astype(int)
    interactions["label"] = interactions["label"].astype("float32")
    interactions = interactions[
        interactions["query_id"].isin(query_id_to_idx)
        & interactions["candidate_id"].isin(candidate_id_to_idx)
    ]

    if sort_by_event_date and "event_date" in interactions.columns:
        interactions = interactions.sort_values("event_date")

    return interactions.reset_index(drop=True)


def prepare_retrieval_pairs(
    interactions_df: pd.DataFrame,
    *,
    query_id_to_idx: dict[int, int],
    candidate_id_to_idx: dict[int, int],
    config: _Config,
    split_name: str,
) -> pd.DataFrame:
    """Filter to positive interactions only."""
    filtered_interactions = filter_and_sample_interactions(
        interactions_df,
        query_id_to_idx=query_id_to_idx,
        candidate_id_to_idx=candidate_id_to_idx,
        config=config,
    )
    positive_interactions = filtered_interactions[filtered_interactions["label"] == 1.0].copy()
    if positive_interactions.empty:
        raise ValueError(f"{split_name} split has no positive interactions for retrieval training.")

    return positive_interactions.reset_index(drop=True)


def prepare_evaluation_inputs(
    X_test: pd.DataFrame,
    query_col: str = "query_id",
    candidate_col: str = "candidate_id",
) -> pd.DataFrame:
    """Validate and normalize evaluation DataFrame to (event_date, query_id, candidate_id, label) format."""
    if not isinstance(X_test, pd.DataFrame):
        raise TypeError(
            "Evaluation features must be a pandas DataFrame with "
            f"'{query_col}' and '{candidate_col}' columns."
        )

    required_columns = {query_col, candidate_col}
    missing_columns = required_columns.difference(X_test.columns)
    if missing_columns:
        raise ValueError(
            f"Evaluation features are missing required columns: {sorted(missing_columns)}"
        )

    evaluation_df = X_test.copy()
    evaluation_df = evaluation_df.rename(columns={query_col: "query_id", candidate_col: "candidate_id"})

    if "label" in evaluation_df.columns:
        evaluation_df["query_id"] = evaluation_df["query_id"].astype(int)
        evaluation_df["candidate_id"] = evaluation_df["candidate_id"].astype(int)
        evaluation_df["label"] = evaluation_df["label"].astype("float32")
        if "event_date" not in evaluation_df.columns:
            evaluation_df["event_date"] = pd.Timestamp("1970-01-01")
        else:
            evaluation_df["event_date"] = pd.to_datetime(evaluation_df["event_date"])
        return evaluation_df.loc[:, ["event_date", "query_id", "candidate_id", "label"]]

    if "clicks" not in evaluation_df.columns:
        raise ValueError(
            "Evaluation features must include either a 'label' column or a 'clicks' column."
        )

    if "event_date" not in evaluation_df.columns:
        evaluation_df["event_date"] = pd.Timestamp("1970-01-01")
    return normalize_interactions(evaluation_df)


def normalize_and_filter_interactions(
    interactions_df: pd.DataFrame,
    *,
    query_id_to_idx: dict[int, int],
    candidate_id_to_idx: dict[int, int],
    config: _Config,
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
        query_id_to_idx=query_id_to_idx,
        candidate_id_to_idx=candidate_id_to_idx,
        config=config,
        sort_by_event_date=True,
    )


def build_evaluation_reference_data(
    train_df: pd.DataFrame | None,
    valid_df: pd.DataFrame | None,
) -> tuple[dict[int, set[int]], list[int]]:
    """Return seen_candidates_by_query and popularity-ranked positive item IDs from train/valid data."""
    seen_candidates_by_query: dict[int, set[int]] = {}
    for dataframe in (train_df, valid_df):
        if dataframe is None or dataframe.empty:
            continue
        grouped = dataframe.groupby("query_id")["candidate_id"]
        for query_id, item_ids in grouped:
            seen_candidates_by_query.setdefault(int(query_id), set()).update(
                int(candidate_id) for candidate_id in item_ids.tolist()
            )

    if train_df is None or train_df.empty:
        return seen_candidates_by_query, []

    train_positive_candidate_ids_by_popularity = (
        train_df.loc[train_df["label"] == 1.0, "candidate_id"]
        .astype(int)
        .value_counts()
        .index
        .tolist()
    )
    return seen_candidates_by_query, train_positive_candidate_ids_by_popularity
