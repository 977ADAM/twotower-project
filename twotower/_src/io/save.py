from __future__ import annotations

from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Protocol

import torch

from twotower._src.config import _Config


class _Saveable(Protocol):
    """Minimal model contract required by the checkpoint save module."""

    config: _Config
    query_col: str
    candidate_col: str
    query_id_to_idx: dict[int, int]
    candidate_id_to_idx: dict[int, int]
    idx_to_query_id: list[int]
    idx_to_candidate_id: list[int]
    train_history: list[dict[str, float]]

    def ensure_fitted(self) -> None:
        ...

    def state_dict(self) -> dict[str, torch.Tensor]:
        ...

    def get_seen_candidates_by_query(self) -> dict[int, set[int]]:
        ...

    def get_train_positive_item_ranking(self) -> list[int]:
        ...

    def get_query_feature_metadata_dict(self) -> dict[str, object]:
        ...

    def get_candidate_feature_metadata_dict(self) -> dict[str, object]:
        ...


class TwoTowerModelSaver:
    """Persist a two-tower model checkpoint through a minimal protocol interface."""

    def save_model(self, model: _Saveable, path: str | PathLike[str]) -> Path:
        model.ensure_fitted()
        target_path = self.resolve_checkpoint_path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": asdict(model.config),
            "query_col": model.query_col,
            "candidate_col": model.candidate_col,
            "state_dict": model.state_dict(),
            "query_id_to_idx": model.query_id_to_idx,
            "candidate_id_to_idx": model.candidate_id_to_idx,
            "idx_to_query_id": model.idx_to_query_id,
            "idx_to_candidate_id": model.idx_to_candidate_id,
            "train_history": model.train_history,
            "seen_candidates_by_query": {
                int(query_id): sorted(int(candidate_id) for candidate_id in item_ids)
                for query_id, item_ids in model.get_seen_candidates_by_query().items()
            },
            "train_positive_candidate_ids_by_popularity": [
                int(candidate_id) for candidate_id in model.get_train_positive_item_ranking()
            ],
            "query_feature_metadata": model.get_query_feature_metadata_dict(),
            "candidate_feature_metadata": model.get_candidate_feature_metadata_dict(),
        }
        torch.save(checkpoint, target_path)
        return target_path

    @staticmethod
    def resolve_checkpoint_path(path: str | PathLike[str]) -> Path:
        checkpoint_path = Path(path)
        if not checkpoint_path.name:
            raise ValueError("Checkpoint path must point to a file.")
        return checkpoint_path
