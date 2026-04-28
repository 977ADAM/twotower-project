import torch
import torch.nn as nn
import torch.nn.functional as F

from twotower.src.backend.config import TwoTowerConfig
from .item_tower import ItemTower
from .user_tower import UserTower


class TwoTowerBase(nn.Module):
    def __init__(self, config: TwoTowerConfig):
        super().__init__()
        self.config = config
        self.user_tower: UserTower | None = None
        self.item_tower: ItemTower | None = None

    def encode_users(self, user_input: torch.Tensor) -> torch.Tensor:
        if self.user_tower is None:
            raise RuntimeError("Model towers are not initialized. Call fit() or load_model() first.")
        return F.normalize(self.user_tower(user_input), dim=-1)

    def encode_items(self, item_input: torch.Tensor) -> torch.Tensor:
        if self.item_tower is None:
            raise RuntimeError("Model towers are not initialized. Call fit() or load_model() first.")
        return F.normalize(self.item_tower(item_input), dim=-1)

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        return (self.encode_users(user_input) * self.encode_items(item_input)).sum(dim=-1)

    def retrieval_logits(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        return torch.matmul(
            self.encode_users(user_input),
            self.encode_items(item_input).T,
        ) / self.config.retrieval_temperature
