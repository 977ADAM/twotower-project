from rich.console import Console

from src.data import fit_id_mappings, load_data, prepare_interactions, split_interactions
from twotower import TwoTower, TwoTowerConfig

console = Console()


def main():
    data_config = {
        "users_path": "data/raw/users.csv",
        "items_path": "data/raw/banners.csv",
        "interactions_path": "data/raw/interactions.csv",
        "max_samples": 250_000,
        "seed": 42,
    }
    model_config = TwoTowerConfig(
        user_embedding_dim=64,
        item_embedding_dim=64,
        hidden_dim=64,
        epochs=10,
        batch_size=2048,
        validation_ratio=0.2,
        test_ratio=0.1,
        max_samples=250_000,
        top_k=5,
    )

    users_df, items_df, interactions_df = load_data(data_config)
    user_idx, item_idx = fit_id_mappings(users_df, items_df)
    interactions = prepare_interactions(interactions_df, user_idx, item_idx, data_config)
    train_df, valid_df, test_df = split_interactions(
        interactions,
        validation_ratio=model_config.validation_ratio,
        test_ratio=model_config.test_ratio,
    )

    model = TwoTower(model_config)
    model.fit(train_df, valid_df)

    metrics = model.evaluate(test_df)
    console.print({"metrics": metrics})

    sample_users = users_df["user_id"].head(3).astype(int).tolist()
    predictions = model.predict(user_ids=sample_users, top_k=model_config.top_k)
    console.print({"sample_predictions": predictions})

    model.save_model("artifacts/twotower_model.pth")


if __name__ == "__main__":
    main()
