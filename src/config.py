from dataclasses import dataclass

@dataclass
class Config:
    users_path: str = "data/raw/users.csv"
    items_path: str = "data/raw/banners.csv"
    interactions_path: str = "data/raw/interactions.csv"
    model_save_path: str = "artifacts/twotower_model.pth"
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    max_samples: int | None = 250_000
    top_k: int = 100
    sample_user_count: int = 3
    sample_prediction_top_k: int = 5
    exclude_seen_predictions: bool = True
    seed: int = 42
