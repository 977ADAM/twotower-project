import pandas as pd
from rich.console import Console

console = Console()

def load_data(config):
    users_df = pd.read_csv(config["users_path"])
    console.print("Users data loaded.")
    items_df = pd.read_csv(config["items_path"])
    console.print("Banners data loaded.")
    interactions_df = pd.read_csv(config["interactions_path"])
    console.print("Interactions data loaded.")
    return users_df, items_df, interactions_df

def fit_id_mappings(users_df: pd.DataFrame, items_df: pd.DataFrame) -> tuple[dict[int, int], dict[int, int]]:
    user_id = users_df["user_id"].astype(int).drop_duplicates().sort_values().tolist()
    console.print(f"Unique users: {len(user_id)}")

    item_id = items_df["banner_id"].astype(int).drop_duplicates().sort_values().tolist()
    console.print(f"Unique items: {len(item_id)}")

    user_idx = {user_id: idx for idx, user_id in enumerate(user_id)}
    console.print("User ID mapping created. ")
    console.print(f"Sample user mapping: {list(user_idx.items())[:5]}")

    item_idx = {item_id: idx for idx, item_id in enumerate(item_id)}
    console.print("Item ID mapping created.")
    console.print(f"Sample item mapping: {list(item_idx.items())[:5]}")

    return user_idx, item_idx

def prepare_interactions(
        interactions_df: pd.DataFrame,
        user_idx: dict[int, int],
        item_idx: dict[int, int],
        config: dict[str, int | None]
    ) -> pd.DataFrame:

    required_columns = {"event_date", "user_id", "banner_id", "clicks"}
    missing_columns = required_columns.difference(interactions_df.columns)
    if missing_columns:
        raise ValueError(f"Interactions dataframe is missing columns: {sorted(missing_columns)}")

    interactions = interactions_df.loc[:, ["event_date", "user_id", "banner_id", "clicks"]].copy()
    interactions["event_date"] = pd.to_datetime(interactions["event_date"])
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["banner_id"] = interactions["banner_id"].astype(int)
    interactions["label"] = (interactions["clicks"] > 0).astype("float32")
    interactions = interactions[
        interactions["user_id"].isin(user_idx)
        & interactions["banner_id"].isin(item_idx)
    ]

    if config["max_samples"] and len(interactions) > config["max_samples"]:
        positives = interactions[interactions["label"] == 1.0]
        negatives = interactions[interactions["label"] == 0.0]
        positive_target = min(len(positives), config["max_samples"] // 2)
        negative_target = min(len(negatives), config["max_samples"] - positive_target)

        sampled_frames = []
        if positive_target:
            sampled_frames.append(
                positives.sample(n=positive_target, random_state=config["seed"], replace=False)
            )
        if negative_target:
            sampled_frames.append(
                negatives.sample(n=negative_target, random_state=config["seed"], replace=False)
            )
        interactions = pd.concat(sampled_frames, ignore_index=True)

    return interactions.sort_values("event_date").reset_index(drop=True)

def split_interactions(
    interactions_df: pd.DataFrame,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if interactions_df.empty:
        raise ValueError("Interactions dataframe must not be empty.")
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1).")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio must be in [0, 1).")
    if validation_ratio + test_ratio >= 1:
        raise ValueError("validation_ratio + test_ratio must be less than 1.")

    total_rows = len(interactions_df)
    train_end = int(total_rows * (1 - validation_ratio - test_ratio))
    valid_end = int(total_rows * (1 - test_ratio))

    train_end = min(max(train_end, 1), total_rows - 2)
    valid_end = min(max(valid_end, train_end + 1), total_rows - 1)

    train_df = interactions_df.iloc[:train_end].copy()
    valid_df = interactions_df.iloc[train_end:valid_end].copy()
    test_df = interactions_df.iloc[valid_end:].copy()
    return train_df, valid_df, test_df



if __name__ == "__main__":
    config = {
        "users_path": "data/raw/users.csv",
        "items_path": "data/raw/banners.csv",
        "interactions_path": "data/raw/interactions.csv",
        "max_samples": 250_000,
        "seed": 42,
    }

    users_df, items_df, interactions_df = load_data(config)

    user_idx, item_idx = fit_id_mappings(users_df, items_df)

    interactions = prepare_interactions(interactions_df, user_idx, item_idx, config)
    console.print(f"Prepared interactions: {len(interactions)}")

    train_df, valid_df, test_df = split_interactions(interactions)
    console.print(f"Training interactions: {len(train_df)}")
    console.print(f"Validation interactions: {len(valid_df)}")
    console.print(f"Test interactions: {len(test_df)}")


    
