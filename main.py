from rich.console import Console

from src.data import load_data
from twotower import TwoTower, TwoTowerConfig

console = Console()


def main():
    data_config = {
        "users_path": "data/raw/users.csv",
        "items_path": "data/raw/banners.csv",
        "interactions_path": "data/raw/interactions.csv",
    }
    model_config = TwoTowerConfig(
        user_embedding_dim=64,
        item_embedding_dim=64,
        hidden_dim=64,
        epochs=3,
        batch_size=2048,
        max_samples=250_000,
        top_k=5,
    )

    users_df, items_df, interactions_df = load_data(data_config)

    model = TwoTower(model_config)
    model.fit(users_df, items_df, interactions_df)

    metrics = model.evaluate()
    console.print({"metrics": metrics})

    sample_users = users_df["user_id"].head(3).astype(int).tolist()
    predictions = model.predict(user_ids=sample_users, top_k=model_config.top_k)
    console.print({"sample_predictions": predictions})

    model.save_model("artifacts/twotower_model.pth")


if __name__ == "__main__":
    main()
