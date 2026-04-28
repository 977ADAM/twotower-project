from dataclasses import dataclass


@dataclass(frozen=True)
class TwoTowerConfig:
    user_embedding_dim: int = 64
    item_embedding_dim: int = 64
    side_feature_embedding_dim: int = 8
    hidden_dim: int = 64
    retrieval_temperature: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 2048
    epochs: int = 25
    max_samples: int | None = 250_000
    eval_top_ks: tuple[int, ...] = (50, 100, 300)
    max_eval_users: int = 500
    top_k: int = 100
    eval_during_training: bool = True
    seed: int = 42
    device: str | None = "cpu"

    def __post_init__(self) -> None:
        if self.user_embedding_dim <= 0:
            raise ValueError(f"`user_embedding_dim` must be positive, got {self.user_embedding_dim}.")
        if self.item_embedding_dim <= 0:
            raise ValueError(f"`item_embedding_dim` must be positive, got {self.item_embedding_dim}.")
        if self.side_feature_embedding_dim <= 0:
            raise ValueError(f"`side_feature_embedding_dim` must be positive, got {self.side_feature_embedding_dim}.")
        if self.hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {self.hidden_dim}.")
        if self.retrieval_temperature <= 0:
            raise ValueError(f"`retrieval_temperature` must be positive, got {self.retrieval_temperature}.")
        if self.learning_rate < 0:
            raise ValueError(f"`learning_rate` must be >= 0, got {self.learning_rate}.")
        if self.batch_size <= 0:
            raise ValueError(f"`batch_size` must be a positive integer, got {self.batch_size}.")
        if self.epochs <= 0:
            raise ValueError(f"`epochs` must be a positive integer, got {self.epochs}.")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError(f"`max_samples` must be a positive integer or None, got {self.max_samples}.")
        if not self.eval_top_ks:
            raise ValueError("`eval_top_ks` must be a non-empty tuple.")
        if any(k <= 0 for k in self.eval_top_ks):
            raise ValueError(f"`eval_top_ks` must contain only positive integers, got {self.eval_top_ks}.")
        if self.max_eval_users <= 0:
            raise ValueError(f"`max_eval_users` must be a positive integer, got {self.max_eval_users}.")
        if self.top_k <= 0:
            raise ValueError(f"`top_k` must be a positive integer, got {self.top_k}.")
        if self.device not in ("cpu", "cuda", None):
            raise ValueError(f"`device` must be 'cpu', 'cuda', or None, got {self.device!r}.")
