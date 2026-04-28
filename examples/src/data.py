import pandas as pd
from rich.console import Console

from examples.src.config import Config
from twotower import split_interactions

console = Console()


def bucketize_age(values: pd.Series) -> pd.Series:
    """Convert a numeric age series into string bucket labels."""
    numeric = pd.to_numeric(values, errors="coerce")
    bucketed = pd.cut(
        numeric,
        bins=[-1, 24, 34, 44, 54, float("inf")],
        labels=["18_24", "25_34", "35_44", "45_54", "55_plus"],
    )
    return bucketed.astype(str).fillna("__unk__")


def _normalize_interactions(interactions_df: pd.DataFrame) -> pd.DataFrame:
    df = interactions_df.loc[:, ["event_date", "user_id", "banner_id", "clicks"]].copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    df["user_id"] = df["user_id"].astype(int)
    df["banner_id"] = df["banner_id"].astype(int)
    df["label"] = (df["clicks"] > 0).astype("float32")
    return df.sort_values("event_date").reset_index(drop=True)


def load_data(config: Config):
    users_df = pd.read_csv(config.users_path)
    console.print("Users data loaded.")
    items_df = pd.read_csv(config.items_path)
    console.print("Banners data loaded.")
    interactions_df = pd.read_csv(config.interactions_path)
    console.print("Interactions data loaded.")
    return users_df, items_df, interactions_df


def prepare_interactions(
    interactions_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> pd.DataFrame:
    known_user_ids = set(users_df["user_id"].astype(int).tolist())
    known_item_ids = set(items_df["banner_id"].astype(int).tolist())
    console.print(f"Unique users: {len(known_user_ids)}")
    console.print(f"Unique items: {len(known_item_ids)}")

    df = _normalize_interactions(interactions_df)
    df = df[df["user_id"].isin(known_user_ids) & df["banner_id"].isin(known_item_ids)]
    console.print(f"Prepared interactions: {len(df)}")
    return df.reset_index(drop=True)


def load_training_frames(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users_df, items_df, interactions_df = load_data(config)
    prepared = prepare_interactions(interactions_df, users_df, items_df)
    train_df, valid_df, test_df = split_interactions(
        prepared,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
    )
    console.print(f"Training interactions: {len(train_df)}")
    console.print(f"Validation interactions: {len(valid_df)}")
    console.print(f"Test interactions: {len(test_df)}")
    return users_df, items_df, train_df, valid_df, test_df


if __name__ == "__main__":
    config = Config()
    load_training_frames(config)
