from dataclasses import dataclass, field
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).parent.parent


@dataclass
class Config:
    users_path: str = field(default_factory=lambda: str(_EXAMPLES_DIR / "data/raw/users.csv"))
    items_path: str = field(default_factory=lambda: str(_EXAMPLES_DIR / "data/raw/banners.csv"))
    interactions_path: str = field(default_factory=lambda: str(_EXAMPLES_DIR / "data/raw/interactions.csv"))
    model_save_path: str = field(default_factory=lambda: str(_EXAMPLES_DIR / "artifacts/twotower_model.pth"))
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    top_k: int = 100
    sample_user_count: int = 3
    sample_prediction_top_k: int = 5
    exclude_seen_predictions: bool = True
    seed: int = 42
