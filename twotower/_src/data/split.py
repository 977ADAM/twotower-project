from __future__ import annotations

import pandas as pd


def normalize_interactions(
    interactions_df: pd.DataFrame,
    query_col: str = "query_id",
    candidate_col: str = "candidate_id",
) -> pd.DataFrame:
    required_columns = {"event_date", query_col, candidate_col, "clicks"}
    missing_columns = required_columns.difference(interactions_df.columns)
    if missing_columns:
        raise ValueError(f"Interactions dataframe is missing columns: {sorted(missing_columns)}")

    interactions = interactions_df.loc[:, ["event_date", query_col, candidate_col, "clicks"]].copy()
    interactions["event_date"] = pd.to_datetime(interactions["event_date"])
    interactions[query_col] = interactions[query_col].astype(int)
    interactions[candidate_col] = interactions[candidate_col].astype(int)
    interactions["label"] = (interactions["clicks"] > 0).astype("float32")
    return interactions.sort_values("event_date").reset_index(drop=True)


def prepare_interactions(
    interactions_df: pd.DataFrame,
    query_id_to_idx: dict[int, int],
    candidate_id_to_idx: dict[int, int],
) -> pd.DataFrame:
    interactions = normalize_interactions(interactions_df)
    interactions = interactions[
        interactions["query_id"].isin(query_id_to_idx)
        & interactions["candidate_id"].isin(candidate_id_to_idx)
    ]
    return interactions.sort_values("event_date").reset_index(drop=True)


def split_interactions(
    interactions_df: pd.DataFrame,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split an interactions DataFrame into train, validation, and test sets.

    The split is **temporal**: all interactions from a date are kept together,
    and dates are assigned to the split whose row-count target they best match.
    This prevents future leakage into the training set.

    Args:
        interactions_df: DataFrame with at least an ``event_date`` column.
            All other columns are preserved unchanged.
        validation_ratio: Approximate fraction of rows to assign to validation.
            Must be in ``[0, 1)``.
        test_ratio: Approximate fraction of rows to assign to test.
            Must be in ``[0, 1)``.

    Returns:
        A 3-tuple ``(train_df, valid_df, test_df)``, each sorted by
        ``event_date`` with the original columns intact.

    Raises:
        ValueError: If the DataFrame is empty, missing ``event_date``, has
            fewer than 3 unique dates, or if the ratio arguments are invalid.

    Example:
        ```python
        train_df, valid_df, test_df = split_interactions(
            interactions_df,
            validation_ratio=0.15,
            test_ratio=0.10,
        )
        ```
    """
    if interactions_df.empty:
        raise ValueError("Interactions dataframe must not be empty.")
    if "event_date" not in interactions_df.columns:
        raise ValueError("Interactions dataframe must contain 'event_date'.")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1).")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1.")

    df = interactions_df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df = df.sort_values("event_date").reset_index(drop=True)

    unique_dates = df["event_date"].drop_duplicates().sort_values().tolist()
    total_dates = len(unique_dates)
    if total_dates < 3:
        raise ValueError("Need at least 3 unique event dates to build train/valid/test splits by date.")

    row_counts_by_date = df.groupby("event_date").size().reindex(unique_dates).tolist()
    prefix_counts = [0]
    for count in row_counts_by_date:
        prefix_counts.append(prefix_counts[-1] + int(count))

    total_rows = prefix_counts[-1]
    target_train_rows = total_rows * (1 - validation_ratio - test_ratio)
    target_valid_rows = total_rows * validation_ratio
    target_test_rows = total_rows * test_ratio

    best_score: float | None = None
    best_boundaries: tuple[int, int] | None = None
    for train_end_idx in range(1, total_dates - 1):
        for valid_end_idx in range(train_end_idx + 1, total_dates):
            train_rows = prefix_counts[train_end_idx]
            valid_rows = prefix_counts[valid_end_idx] - prefix_counts[train_end_idx]
            test_rows = prefix_counts[total_dates] - prefix_counts[valid_end_idx]

            if train_rows == 0 or valid_rows == 0 or test_rows == 0:
                continue

            score = (
                (train_rows - target_train_rows) ** 2
                + (valid_rows - target_valid_rows) ** 2
                + (test_rows - target_test_rows) ** 2
            )
            if best_score is None or score < best_score:
                best_score = score
                best_boundaries = (train_end_idx, valid_end_idx)

    if best_boundaries is None:
        raise ValueError("Could not allocate non-empty train/valid/test splits by date.")

    train_end_idx, valid_end_idx = best_boundaries
    train_dates = set(unique_dates[:train_end_idx])
    valid_dates = set(unique_dates[train_end_idx:valid_end_idx])
    test_dates = set(unique_dates[valid_end_idx:])

    train_df = df[df["event_date"].isin(train_dates)].copy()
    valid_df = df[df["event_date"].isin(valid_dates)].copy()
    test_df = df[df["event_date"].isin(test_dates)].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Date-based split produced an empty split.")

    return train_df, valid_df, test_df
