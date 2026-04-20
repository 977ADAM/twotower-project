import pandas as pd
from rich.console import Console
from config import Config
from twotower.data import prepare_interactions as prepare_twotower_interactions
from twotower.data import split_interactions as split_twotower_interactions

console = Console()

def load_data(config: Config):
    users_df = pd.read_csv(config.users_path)
    console.print("Users data loaded.")
    items_df = pd.read_csv(config.items_path)
    console.print("Banners data loaded.")
    interactions_df = pd.read_csv(config.interactions_path)
    console.print("Interactions data loaded.")
    return users_df, items_df, interactions_df

def _build_known_id_mappings(
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    user_ids = users_df["user_id"].astype(int).drop_duplicates().sort_values().tolist()
    item_ids = items_df["banner_id"].astype(int).drop_duplicates().sort_values().tolist()
    console.print(f"Unique users: {len(user_ids)}")
    console.print(f"Unique items: {len(item_ids)}")
    return (
        {user_id: idx for idx, user_id in enumerate(user_ids)},
        {item_id: idx for idx, item_id in enumerate(item_ids)},
    )

def prepare_interactions(
    interactions_df: pd.DataFrame,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    config: Config,
) -> pd.DataFrame:
    user_id_to_idx, item_id_to_idx = _build_known_id_mappings(users_df, items_df)
    prepared_interactions = prepare_twotower_interactions(
        interactions_df=interactions_df,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        max_samples=config.max_samples,
        seed=config.seed,
    )
    console.print(f"Prepared interactions: {len(prepared_interactions)}")
    return prepared_interactions

def split_interactions(
    interactions_df: pd.DataFrame,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return split_twotower_interactions(
        interactions_df=interactions_df,
        validation_ratio=validation_ratio,
        test_ratio=test_ratio,
    )

def load_training_frames(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    users_df, items_df, interactions_df = load_data(config)
    prepared_interactions = prepare_interactions(
        interactions_df=interactions_df,
        users_df=users_df,
        items_df=items_df,
        config=config,
    )
    train_df, valid_df, test_df = split_interactions(
        prepared_interactions,
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


    
