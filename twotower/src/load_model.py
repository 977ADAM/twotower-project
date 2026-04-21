from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Protocol

import torch

from twotower.src.backend.config import TwoTowerConfig
from twotower.src.features import FeatureMetadata


@dataclass(slots=True)
class LoadedCheckpointState:
    """Normalized checkpoint state ready to be applied to a model instance."""

    config: TwoTowerConfig
    device: torch.device
    user_id_to_idx: dict[int, int]
    item_id_to_idx: dict[int, int]
    idx_to_user_id: list[int]
    idx_to_item_id: list[int]
    train_history: list[dict[str, float]]
    seen_items_by_user: dict[int, set[int]]
    train_positive_item_ids_by_popularity: list[int]
    user_feature_metadata: FeatureMetadata
    item_feature_metadata: FeatureMetadata


class LoadableTwoTower(Protocol):
    """Minimal model contract required by the checkpoint load module."""

    def validate_checkpoint(self, checkpoint: object, checkpoint_path: Path) -> None:
        ...

    def resolve_device(self, device: str | None) -> torch.device:
        ...

    def apply_loaded_checkpoint_state(self, state: LoadedCheckpointState) -> None:
        ...

    def build_towers(self, num_users: int, num_items: int) -> None:
        ...

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        ...

    def to(self, device: torch.device):
        ...

    def invalidate_item_embedding_cache(self) -> None:
        ...

    def eval(self):
        ...


class TwoTowerModelLoader:
    """Restore a two-tower model checkpoint through a minimal protocol interface."""

    def load_model(
        self,
        model: LoadableTwoTower,
        path: str | PathLike[str],
    ) -> None:
        checkpoint_path = self.resolve_checkpoint_path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint was not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.validate_checkpoint(checkpoint, checkpoint_path)

        loaded_state = self.build_loaded_checkpoint_state(model, checkpoint)
        model.apply_loaded_checkpoint_state(loaded_state)
        model.build_towers(len(loaded_state.idx_to_user_id), len(loaded_state.idx_to_item_id))
        model.load_state_dict(checkpoint["state_dict"])
        model.to(loaded_state.device)
        model.invalidate_item_embedding_cache()
        model.eval()

    @staticmethod
    def resolve_checkpoint_path(path: str | PathLike[str]) -> Path:
        checkpoint_path = Path(path)
        if not checkpoint_path.name:
            raise ValueError("Checkpoint path must point to a file.")
        return checkpoint_path

    @staticmethod
    def build_loaded_checkpoint_state(
        model: LoadableTwoTower,
        checkpoint: dict[str, object],
    ) -> LoadedCheckpointState:
        config = TwoTowerConfig(**checkpoint["config"])
        return LoadedCheckpointState(
            config=config,
            device=model.resolve_device(config.device),
            user_id_to_idx={
                int(user_id): int(index)
                for user_id, index in dict(checkpoint["user_id_to_idx"]).items()
            },
            item_id_to_idx={
                int(item_id): int(index)
                for item_id, index in dict(checkpoint["item_id_to_idx"]).items()
            },
            idx_to_user_id=[int(user_id) for user_id in checkpoint["idx_to_user_id"]],
            idx_to_item_id=[int(item_id) for item_id in checkpoint["idx_to_item_id"]],
            train_history=[
                {str(metric_name): float(metric_value) for metric_name, metric_value in record.items()}
                for record in checkpoint.get("train_history", [])
            ],
            seen_items_by_user={
                int(user_id): {int(item_id) for item_id in item_ids}
                for user_id, item_ids in dict(checkpoint.get("seen_items_by_user", {})).items()
            },
            train_positive_item_ids_by_popularity=[
                int(item_id)
                for item_id in checkpoint.get("train_positive_item_ids_by_popularity", [])
            ],
            user_feature_metadata=FeatureMetadata.from_dict(
                checkpoint.get("user_feature_metadata")
            ),
            item_feature_metadata=FeatureMetadata.from_dict(
                checkpoint.get("item_feature_metadata")
            ),
        )
