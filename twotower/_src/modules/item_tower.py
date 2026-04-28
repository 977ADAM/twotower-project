import torch
import torch.nn as nn
import torch.nn.functional as F

from twotower.src.backend.config import TwoTowerConfig
from twotower.src.features import FeatureMetadata, FeatureTables


class ItemTower(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        config: TwoTowerConfig,
        feature_tables: FeatureTables | None = None,
        feature_metadata: FeatureMetadata | None = None,
    ):
        super().__init__()
        self.feature_metadata = feature_metadata or FeatureMetadata.empty()
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=config.item_embedding_dim)
        self.scalar_feature_embeddings = nn.ModuleDict(
            {
                feature_name: nn.Embedding(
                    num_embeddings=self.feature_metadata.vocab_sizes[feature_name],
                    embedding_dim=config.side_feature_embedding_dim,
                )
                for feature_name in self.feature_metadata.scalar_feature_names
            }
        )
        self.multi_feature_embeddings = nn.ModuleDict(
            {
                feature_name: nn.Embedding(
                    num_embeddings=self.feature_metadata.vocab_sizes[feature_name],
                    embedding_dim=config.side_feature_embedding_dim,
                )
                for feature_name in self.feature_metadata.multi_feature_names
            }
        )
        self._register_feature_buffers(num_embeddings, feature_tables)

        total_input_dim = config.item_embedding_dim
        total_input_dim += len(self.feature_metadata.scalar_feature_names) * config.side_feature_embedding_dim
        total_input_dim += len(self.feature_metadata.multi_feature_names) * config.side_feature_embedding_dim

        self.fc = nn.Linear(total_input_dim, config.hidden_dim)
        self.norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, item_input: torch.Tensor) -> torch.Tensor:
        feature_parts = [self.embedding(item_input)]

        for feature_name in self.feature_metadata.scalar_feature_names:
            feature_indices = getattr(self, f"scalar_feature_{feature_name}").index_select(0, item_input)
            feature_parts.append(self.scalar_feature_embeddings[feature_name](feature_indices))

        for feature_name in self.feature_metadata.multi_feature_names:
            feature_indices = getattr(self, f"multi_feature_{feature_name}").index_select(0, item_input)
            pooled_embedding = self.multi_feature_embeddings[feature_name](feature_indices).mean(dim=1)
            feature_parts.append(pooled_embedding)

        x = torch.cat(feature_parts, dim=-1)
        x = self.fc(x)
        x = F.relu(x)
        return self.norm(x)

    def _register_feature_buffers(
        self,
        num_entities: int,
        feature_tables: FeatureTables | None,
    ) -> None:
        scalar_features = feature_tables.scalar_features if feature_tables is not None else {}
        multi_features = feature_tables.multi_features if feature_tables is not None else {}

        for feature_name in self.feature_metadata.scalar_feature_names:
            feature_tensor = scalar_features.get(feature_name)
            if feature_tensor is None:
                feature_tensor = torch.zeros(num_entities, dtype=torch.long)
            self.register_buffer(f"scalar_feature_{feature_name}", feature_tensor.clone(), persistent=True)

        for feature_name in self.feature_metadata.multi_feature_names:
            feature_tensor = multi_features.get(feature_name)
            if feature_tensor is None:
                feature_width = self.feature_metadata.multi_feature_widths[feature_name]
                feature_tensor = torch.zeros((num_entities, feature_width), dtype=torch.long)
            self.register_buffer(f"multi_feature_{feature_name}", feature_tensor.clone(), persistent=True)
