import torch
import torch.nn as nn
import torch.nn.functional as F

from twotower.config import TwoTowerConfig
from .item_tower import ItemTower
from .user_tower import UserTower

class TwoTowerBase(nn.Module):
    def __init__(
        self,
        config: TwoTowerConfig,
        num_users: int | None = None,
        num_items: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.user_tower: UserTower | None = None
        self.item_tower: ItemTower | None = None
        if num_users is not None and num_items is not None:
            self.build_towers(num_users, num_items)

    def build_towers(self, num_users: int, num_items: int) -> None:
        self.user_tower = UserTower(num_users, self.config)
        self.item_tower = ItemTower(num_items, self.config)

    def forward(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.user_tower is None or self.item_tower is None:
            raise RuntimeError("Model towers are not initialized. Call fit() or load_model() first.")
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        return user_embedding, item_embedding

    def retrieval_logits(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
    ) -> torch.Tensor:
        user_embedding, item_embedding = self.forward(user_input, item_input)
        user_embedding = F.normalize(user_embedding, dim=-1)
        item_embedding = F.normalize(item_embedding, dim=-1)
        return torch.matmul(user_embedding, item_embedding.T) / self.config.retrieval_temperature

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        user_embedding, item_embedding = self.forward(user_input, item_input)
        user_embedding = F.normalize(user_embedding, dim=-1)
        item_embedding = F.normalize(item_embedding, dim=-1)
        return (user_embedding * item_embedding).sum(dim=-1)
