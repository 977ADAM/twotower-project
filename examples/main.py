import pandas as pd
from rich.console import Console
from src.config import Config
from src.data import bucketize_age, load_training_frames

from twotower import FeatureConfig, MultiFeatureSpec, TwoTower

console = Console()


def main():
    config = Config()
    users_df, items_df, train_df, valid_df, test_df = load_training_frames(config)

    users_df = users_df.copy()
    users_df["age_bucket"] = bucketize_age(users_df["age"])

    items_df = items_df.copy()
    items_df["target_age_bucket"] = bucketize_age(
        (
            pd.to_numeric(items_df["target_age_min"], errors="coerce")
            + pd.to_numeric(items_df["target_age_max"], errors="coerce")
        )
        / 2.0
    )

    user_feature_config = FeatureConfig(
        scalar_features=(
            "age_bucket",
            "gender",
            "city_tier",
            "device_os",
            "platform",
            "income_band",
            "activity_segment",
            "is_premium",
        ),
        multi_features=(
            MultiFeatureSpec("interest_ids", columns=("interest_1", "interest_2", "interest_3")),
        ),
    )
    item_feature_config = FeatureConfig(
        scalar_features=(
            "brand",
            "category",
            "subcategory",
            "banner_format",
            "campaign_goal",
            "target_gender",
            "target_age_bucket",
        ),
    )

    model = TwoTower(tower_dims=(256, 128), dropout=0.3)
    history = model.fit(
        train_df,
        validation_data=valid_df,
        users_df=users_df,
        items_df=items_df,
        user_feature_config=user_feature_config,
        item_feature_config=item_feature_config,
        observed_ratio=0.8,
        weight_decay=1e-3,
        patience=5,
        early_stopping_metric="recall_at_100",
    )
    console.print({"history_tail": history[-3:]})

    metrics = model.evaluate(test_df, top_k=config.top_k)
    console.print({"metrics": metrics})

    sample_users = model.idx_to_user_id[: min(config.sample_user_count, len(model.idx_to_user_id))]
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
