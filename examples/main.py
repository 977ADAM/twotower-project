import pandas as pd
from rich.console import Console
from src.config import Config
from src.data import bucketize_age, load_training_frames

from twotower import TwoTower

console = Console()


def main():
    config = Config()
    queries_df, candidates_df, train_df, valid_df, test_df = load_training_frames(config)

    queries_df = queries_df.copy()
    queries_df["age_bucket"] = bucketize_age(queries_df["age"])

    candidates_df = candidates_df.copy()
    candidates_df["target_age_bucket"] = bucketize_age(
        (
            pd.to_numeric(candidates_df["target_age_min"], errors="coerce")
            + pd.to_numeric(candidates_df["target_age_max"], errors="coerce")
        )
        / 2.0
    )

    model = TwoTower(tower_dims=(256, 128), dropout=0.1, side_feature_embedding_dim=16, hidden_dim=128)
    history = model.fit(
        train_df,
        validation_data=valid_df,
        query_col="user_id",
        candidate_col="banner_id",
        queries_df=queries_df,
        candidates_df=candidates_df,
        query_features=["age_bucket", "gender", "city_tier", "device_os", "platform", "income_band", "activity_segment", "is_premium"],
        query_multi_features={"interest_ids": ["interest_1", "interest_2", "interest_3"]},
        candidate_features=["brand", "category", "subcategory", "banner_format", "campaign_goal", "target_gender", "target_age_bucket"],
        observed_ratio=0.8,
        learning_rate=2e-4,
        weight_decay=1e-4,
        patience=10,
        early_stopping_metric="recall_at_100",
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
