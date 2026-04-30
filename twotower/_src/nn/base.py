import torch
import torch.nn as nn
import torch.nn.functional as F

from twotower._src.config import _Config

from .tower import Tower


class TwoTowerBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config: _Config | None = None
        self.query_tower: Tower | None = None
        self.candidate_tower: Tower | None = None

    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor:
        if self.query_tower is None:
            raise RuntimeError("Model towers are not initialized. Call fit() or load_model() first.")
        return F.normalize(self.query_tower(user_input), dim=-1)

    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor:
        if self.candidate_tower is None:
            raise RuntimeError("Model towers are not initialized. Call fit() or load_model() first.")
        return F.normalize(self.candidate_tower(item_input), dim=-1)

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        return (self.encode_queries(user_input) * self.encode_candidates(item_input)).sum(dim=-1)

    def retrieval_logits(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        if self.config is None:
            raise RuntimeError("Model is not initialized. Call fit() or load_model() first.")
        return torch.matmul(
            self.encode_queries(user_input),
            self.encode_candidates(item_input).T,
        ) / self.config.retrieval_temperature
