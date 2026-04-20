from __future__ import annotations

from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Protocol

import torch

from twotower.config import TwoTowerConfig


class SaveableTwoTower(Protocol):
    """Minimal model contract required by the checkpoint save module."""

    config: TwoTowerConfig
    user_id_to_idx: dict[int, int]
    item_id_to_idx: dict[int, int]
    idx_to_user_id: list[int]
    idx_to_item_id: list[int]
    train_history: list[dict[str, float]]

    def ensure_fitted(self) -> None:
        ...

    def state_dict(self) -> dict[str, torch.Tensor]:
        ...

    def get_seen_items_by_user(self) -> dict[int, set[int]]:
        ...

    def get_train_positive_item_ranking(self) -> list[int]:
        ...

    def get_user_feature_metadata_dict(self) -> dict[str, object]:
        ...

    def get_item_feature_metadata_dict(self) -> dict[str, object]:
        ...


class TwoTowerModelSaver:
    """Persist a two-tower model checkpoint through a minimal protocol interface."""

    def save_model(self, model: SaveableTwoTower, path: str | PathLike[str]) -> Path:
        model.ensure_fitted()
        target_path = self.resolve_checkpoint_path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "user_id_to_idx": model.user_id_to_idx,
            "item_id_to_idx": model.item_id_to_idx,
            "idx_to_user_id": model.idx_to_user_id,
            "idx_to_item_id": model.idx_to_item_id,
            "train_history": model.train_history,
            "seen_items_by_user": {
                int(user_id): sorted(int(item_id) for item_id in item_ids)
                for user_id, item_ids in model.get_seen_items_by_user().items()
            },
            "train_positive_item_ids_by_popularity": [
                int(item_id) for item_id in model.get_train_positive_item_ranking()
            ],
            "user_feature_metadata": model.get_user_feature_metadata_dict(),
            "item_feature_metadata": model.get_item_feature_metadata_dict(),
        }
        torch.save(checkpoint, target_path)
        return target_path

    @staticmethod
    def resolve_checkpoint_path(path: str | PathLike[str]) -> Path:
        checkpoint_path = Path(path)
        if not checkpoint_path.name:
            raise ValueError("Checkpoint path must point to a file.")
        return checkpoint_path
