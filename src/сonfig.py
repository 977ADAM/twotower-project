from dataclasses import dataclass


@dataclass
class Config:
    users_path: str = "data/raw/users.csv"
    items_path: str = "data/raw/banners.csv"
    interactions_path: str = "data/raw/interactions.csv"
    model_save_path: str = "artifacts/twotower_model.pth"
    max_samples: int | None = 250_000
    seed: int = 42