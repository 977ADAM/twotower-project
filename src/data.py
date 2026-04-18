import pandas as pd
from rich.console import Console
from src.сonfig import Config
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
        config: Config,
    ) -> pd.DataFrame:
    return prepare_twotower_interactions(
        interactions_df=interactions_df,
        user_id_to_idx=user_idx,
        item_id_to_idx=item_idx,
        max_samples=config.max_samples,
        seed=config.seed,
    )

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



if __name__ == "__main__":
    config = Config()

    users_df, items_df, interactions_df = load_data(config)

    user_idx, item_idx = fit_id_mappings(users_df, items_df)

    interactions = prepare_interactions(interactions_df, user_idx, item_idx, config)
    console.print(f"Prepared interactions: {len(interactions)}")

    train_df, valid_df, test_df = split_interactions(interactions)
    console.print(f"Training interactions: {len(train_df)}")
    console.print(f"Validation interactions: {len(valid_df)}")
    console.print(f"Test interactions: {len(test_df)}")


    
