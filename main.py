from rich.console import Console

from src.data import fit_id_mappings, load_data, prepare_interactions, split_interactions
from src.сonfig import Config
from twotower import TwoTower


console = Console()


def main():
    config = Config()
    users_df, items_df, interactions_df = load_data(config)
    user_idx, item_idx = fit_id_mappings(users_df, items_df)
    interactions = prepare_interactions(interactions_df, user_idx, item_idx, config)
    train_df, valid_df, test_df = split_interactions(interactions)

    model = TwoTower()
    model.fit(train_df, valid_df)

    metrics = model.evaluate(test_df)
    console.print({"metrics": metrics})

    sample_users = users_df["user_id"].head(3).astype(int).tolist()
    predictions = model.predict(user_ids=sample_users)
    console.print({"sample_predictions": predictions})

    model.save_model(config.model_save_path)


if __name__ == "__main__":
    main()
