import json
import os
from dataclasses import dataclass

from twotower.src.api_export import twotower_export

_MAX_EPOCHS = None

@dataclass
class TwoTowerConfig:
    user_embedding_dim: int = 64
    item_embedding_dim: int = 64
    side_feature_embedding_dim: int = 8
    hidden_dim: int = 64
    retrieval_temperature: float = 0.1
    symmetric_retrieval_loss: bool = True
    learning_rate: float = 1e-3
    batch_size: int = 2048
    epochs: int = 25
    validation_ratio: float = 0.2
    test_ratio: float = 0.1
    max_samples: int | None = 250_000
    eval_top_ks: tuple[int, ...] = (50, 100, 300)
    max_eval_users: int = 500
    top_k: int = 100
    eval_during_training: bool = True
    seed: int = 42
    device: str | None = "cpu"


@twotower_export(["twotower.config.max_epochs"])
def max_epochs():
    return _MAX_EPOCHS
