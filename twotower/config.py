from dataclasses import dataclass

@dataclass
class TwoTowerConfig:
    user_embedding_dim: int = 64
    item_embedding_dim: int = 64
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    batch_size: int = 2048
    epochs: int = 3
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    max_samples: int | None = 250_000
    max_eval_users: int = 500
    top_k: int = 10
    seed: int = 42
    device: str | None = None
