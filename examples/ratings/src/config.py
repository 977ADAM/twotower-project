from dataclasses import dataclass, field
from pathlib import Path

_RATINGS_DIR = Path(__file__).parent.parent
_EXAMPLES_DIR = _RATINGS_DIR.parent


@dataclass
class Config:
    ratings_path: str = field(default_factory=lambda: str(_EXAMPLES_DIR / "data/raw/ratings.csv"))
    movies_path: str = field(default_factory=lambda: str(_EXAMPLES_DIR / "data/raw/movies.csv"))
    model_save_path: str = field(default_factory=lambda: str(_RATINGS_DIR / "artifacts/ratings_model.pth"))
    sample_users: int = 500
    positive_threshold: float = 4.0
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    top_k: int = 50
    sample_user_count: int = 3
    sample_prediction_top_k: int = 5
    exclude_seen_predictions: bool = True
    seed: int = 42
