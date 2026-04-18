import torch
import torch.nn as nn
import torch.nn.functional as F

from twotower.config import TwoTowerConfig


class UserTower(nn.Module):
    def __init__(self, num_emb: int, config: TwoTowerConfig):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=num_emb,embedding_dim=config.user_embedding_dim)
        self.fc = nn.Linear(config.user_embedding_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, user_input: torch.Tensor) -> torch.Tensor:
        x = self.emb(user_input)
        x = self.fc(x)
        x = F.relu(x)
        return self.norm(x)