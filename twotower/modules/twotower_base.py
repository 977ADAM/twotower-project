import torch
import torch.nn as nn
import torch.nn.functional as F

from twotower.config import TwoTowerConfig
from .item_tower import ItemTower
from .user_tower import UserTower


class TwoTowerBase(nn.Module):
    def __init__(self, config: TwoTowerConfig):
        super().__init__()
        self.config = config
        self.user_tower: UserTower | None = None
        self.item_tower: ItemTower | None = None

    def forward(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.user_tower is None or self.item_tower is None:
            raise RuntimeError("Model towers are not initialized. Call fit() or load_model() first.")
        return self.user_tower(user_input), self.item_tower(item_input)

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.forward(user_input, item_input)
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)
        return (user_emb * item_emb).sum(dim=-1)

    def retrieval_logits(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        user_emb, item_emb = self.forward(user_input, item_input)
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)
        return torch.matmul(user_emb, item_emb.T) / self.config.retrieval_temperature
