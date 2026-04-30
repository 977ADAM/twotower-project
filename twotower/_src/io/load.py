from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

import torch

from twotower._src.config import _Config
from twotower._src.data.features import FeatureMetadata


@dataclass(slots=True)
class LoadedCheckpointState:
    """Normalized checkpoint state ready to be applied to a model instance."""

    config: _Config
    query_col: str
    candidate_col: str
    device: torch.device
    query_id_to_idx: dict[int, int]
    candidate_id_to_idx: dict[int, int]
    idx_to_query_id: list[int]
    idx_to_candidate_id: list[int]
    train_history: list[dict[str, float]]
    seen_candidates_by_query: dict[int, set[int]]
    train_positive_candidate_ids_by_popularity: list[int]
    query_feature_metadata: FeatureMetadata
    candidate_feature_metadata: FeatureMetadata


class _Loadable(Protocol):
    """Minimal model contract required by the checkpoint load module."""

    def validate_checkpoint(self, checkpoint: object, checkpoint_path: Path) -> None:
        ...

    def resolve_device(self, device: str | None) -> torch.device:
        ...

    def apply_loaded_checkpoint_state(self, state: LoadedCheckpointState) -> None:
        ...

    def build_towers(self, num_users: int, num_items: int) -> None:
        ...

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        ...

    def to(self, device: torch.device) -> object:
        ...

    def invalidate_item_embedding_cache(self) -> None:
        ...

    def eval(self) -> object:
        ...


class TwoTowerModelLoader:
    """Restore a two-tower model checkpoint through a minimal protocol interface."""

    def load_model(
        self,
        model: _Loadable,
        path: str | PathLike[str],
    ) -> None:
        checkpoint_path = self.resolve_checkpoint_path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint was not found: {checkpoint_path}")

        checkpoint: dict[str, Any] = torch.load(checkpoint_path, map_location="cpu")
        model.validate_checkpoint(checkpoint, checkpoint_path)

        loaded_state = self.build_loaded_checkpoint_state(model, checkpoint)
        model.apply_loaded_checkpoint_state(loaded_state)
        model.build_towers(len(loaded_state.idx_to_query_id), len(loaded_state.idx_to_candidate_id))
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
        model: _Loadable,
        checkpoint: dict[str, Any],
    ) -> LoadedCheckpointState:
        known_fields = {f.name for f in dataclasses.fields(_Config)}
        config_dict = {k: v for k, v in dict(checkpoint["config"]).items() if k in known_fields}
        config = _Config(**config_dict)
        return LoadedCheckpointState(
            config=config,
            query_col=str(checkpoint.get("query_col", "query_id")),
            candidate_col=str(checkpoint.get("candidate_col", "candidate_id")),
            device=model.resolve_device(config.device),
            query_id_to_idx={
                int(query_id): int(index)
                for query_id, index in dict(checkpoint["query_id_to_idx"]).items()
            },
            candidate_id_to_idx={
                int(candidate_id): int(index)
                for candidate_id, index in dict(checkpoint["candidate_id_to_idx"]).items()
            },
            idx_to_query_id=[int(query_id) for query_id in checkpoint["idx_to_query_id"]],
            idx_to_candidate_id=[int(candidate_id) for candidate_id in checkpoint["idx_to_candidate_id"]],
            train_history=[
                {str(metric_name): float(metric_value) for metric_name, metric_value in record.items()}
                for record in checkpoint.get("train_history", [])
            ],
            seen_candidates_by_query={
                int(query_id): {int(candidate_id) for candidate_id in item_ids}
                for query_id, item_ids in dict(checkpoint.get("seen_candidates_by_query", {})).items()
            },
            train_positive_candidate_ids_by_popularity=[
                int(candidate_id)
                for candidate_id in checkpoint.get("train_positive_candidate_ids_by_popularity", [])
            ],
            query_feature_metadata=FeatureMetadata.from_dict(
                checkpoint.get("query_feature_metadata")
            ),
            candidate_feature_metadata=FeatureMetadata.from_dict(
                checkpoint.get("candidate_feature_metadata")
            ),
        )
