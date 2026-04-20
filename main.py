from rich.console import Console

from src.data import load_training_frames
from src.сonfig import Config
from twotower import TwoTower, TwoTowerConfig

console = Console()


def main():
    config = Config()
    users_df, items_df, train_df, valid_df, test_df = load_training_frames(config)

    model = TwoTower(
        TwoTowerConfig(
            max_samples=config.max_samples,
            top_k=config.top_k,
            seed=config.seed,
        )
    )
    history = model.fit(
        X_train=train_df.loc[:, ["user_id", "banner_id"]].copy(),
        y_train=train_df["label"].copy(),
        X_valid=valid_df.loc[:, ["user_id", "banner_id"]].copy(),
        y_valid=valid_df["label"].copy(),
        users_df=users_df,
        items_df=items_df,
    )
    console.print({"history_tail": history[-3:]})

    metrics = model.evaluate(test_df, top_k=config.top_k)
    console.print({"metrics": metrics})

    sample_users = model.idx_to_user_id[: min(config.sample_user_count, len(model.idx_to_user_id))]
    predictions = model.predict(
        user_ids=sample_users,
        top_k=config.sample_prediction_top_k,
        exclude_seen=config.exclude_seen_predictions,
        strict=True,
    )
    console.print({"sample_predictions": predictions})

    model.save_model(config.model_save_path)


if __name__ == "__main__":
    main()
