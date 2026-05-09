import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from src.config import Config
from src.data import load_training_frames

from twotower import TwoTower

console = Console()


def main() -> None:
    config = Config()
    movies_df, train_df, valid_df, test_df = load_training_frames(config)

    model = TwoTower(tower_dims=(128, 64), dropout=0.1, hidden_dim=64)
    history = model.fit(
        train_df,
        validation_data=valid_df,
        query_col="userId",
        candidate_col="movieId",
        candidates_df=movies_df,
        candidate_multi_features={"genres": ["genre_1", "genre_2", "genre_3"]},
        observed_ratio=0.8,
        learning_rate=1e-3,
        patience=5,
        epochs=5,
        early_stopping_metric="recall_at_50",
        eval_top_ks=(50,),
        top_k=50,
    )
    
    console.print({"history_tail": history[-3:]})

    metrics = model.evaluate(test_df, top_k=config.top_k)
    console.print({"metrics": metrics})

    sample_users = model.idx_to_query_id[: min(config.sample_user_count, len(model.idx_to_query_id))]
    predictions = model.retrieve(
        user_ids=sample_users,
        top_k=config.sample_prediction_top_k,
        exclude_seen=config.exclude_seen_predictions,
        strict=True,
    )
    console.print(predictions.head(20).to_string())

    model.save_model(config.model_save_path)


if __name__ == "__main__":
    main()
