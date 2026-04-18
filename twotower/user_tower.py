import torch
import torch.nn as nn
import torch.nn.functional as F

from twotower.config import TwoTowerConfig


class UserTower(nn.Module):
    def __init__(self, num_embeddings: int, config: TwoTowerConfig):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=config.user_embedding_dim,
        )
        self.fc = nn.Linear(config.user_embedding_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, user_input: torch.Tensor) -> torch.Tensor:
        x = self.embedding(user_input)
        x = self.fc(x)
        x = F.relu(x)
        return self.norm(x)