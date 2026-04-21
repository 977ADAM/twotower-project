from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from twotower.src.backend.config import TwoTowerConfig
from twotower.features import FeatureMetadata
from twotower.load_model import LoadedCheckpointState, TwoTowerModelLoader
from twotower.save_model import TwoTowerModelSaver


class StubSaveableModel:
    def __init__(self):
        self.config = TwoTowerConfig(top_k=7, device="cpu")
        self.user_id_to_idx = {1: 0}
        self.item_id_to_idx = {10: 0, 30: 1}
        self.idx_to_user_id = [1]
        self.idx_to_item_id = [10, 30]
        self.train_history = [{"epoch": 1.0, "train_loss": 0.5, "valid_loss": 0.4}]
        self.ensure_fitted_calls = 0

    def ensure_fitted(self) -> None:
        self.ensure_fitted_calls += 1

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": torch.tensor([1.0, 2.0])}

    def get_seen_items_by_user(self) -> dict[int, set[int]]:
        return {1: {30, 10}}

    def get_train_positive_item_ranking(self) -> list[int]:
        return [30, 10]

    def get_user_feature_metadata_dict(self) -> dict[str, object]:
        return FeatureMetadata.empty().to_dict()

    def get_item_feature_metadata_dict(self) -> dict[str, object]:
        return FeatureMetadata.empty().to_dict()


class StubLoadableModel:
    def __init__(self):
        self.validate_calls: list[tuple[object, Path]] = []
        self.resolve_device_calls: list[str | None] = []
        self.applied_state: LoadedCheckpointState | None = None
        self.build_tower_calls: list[tuple[int, int]] = []
        self.loaded_state_dict: dict[str, torch.Tensor] | None = None
        self.to_calls: list[torch.device] = []
        self.invalidate_calls = 0
        self.eval_calls = 0

    def validate_checkpoint(self, checkpoint: object, checkpoint_path: Path) -> None:
        self.validate_calls.append((checkpoint, checkpoint_path))
        if not isinstance(checkpoint, dict):
            raise ValueError("invalid checkpoint")

    def resolve_device(self, device: str | None) -> torch.device:
        self.resolve_device_calls.append(device)
        return torch.device("cpu")

    def apply_loaded_checkpoint_state(self, state: LoadedCheckpointState) -> None:
        self.applied_state = state

    def build_towers(self, num_users: int, num_items: int) -> None:
        self.build_tower_calls.append((num_users, num_items))

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        self.loaded_state_dict = state_dict

    def to(self, device: torch.device):
        self.to_calls.append(device)
        return self

    def invalidate_item_embedding_cache(self) -> None:
        self.invalidate_calls += 1

    def eval(self):
        self.eval_calls += 1


def test_save_model_persists_checkpoint_payload():
    saver = TwoTowerModelSaver()
    model = StubSaveableModel()

    with tempfile.TemporaryDirectory() as tmpdir:
        target_path = Path(tmpdir) / "nested" / "model.pth"
        saved_path = saver.save_model(model, target_path)
        checkpoint = torch.load(saved_path, map_location="cpu")

    assert model.ensure_fitted_calls == 1
    assert saved_path == target_path
    assert "config" in checkpoint
    assert torch.equal(checkpoint["state_dict"]["weight"], torch.tensor([1.0, 2.0]))
    assert checkpoint["seen_items_by_user"] == {1: [10, 30]}
    assert checkpoint["train_positive_item_ids_by_popularity"] == [30, 10]


def test_load_model_restores_checkpoint_state_through_protocol():
    loader = TwoTowerModelLoader()
    model = StubLoadableModel()
    checkpoint = {
        "config": TwoTowerConfig(top_k=5, device="cpu").__dict__,
        "state_dict": {"weight": torch.tensor([3.0])},
        "user_id_to_idx": {1: 0},
        "item_id_to_idx": {10: 0, 30: 1},
        "idx_to_user_id": [1],
        "idx_to_item_id": [10, 30],
        "train_history": [{"epoch": 1.0, "train_loss": 0.2, "valid_loss": 0.1}],
        "seen_items_by_user": {1: [10, 30]},
        "train_positive_item_ids_by_popularity": [30, 10],
        "user_feature_metadata": FeatureMetadata.empty().to_dict(),
        "item_feature_metadata": FeatureMetadata.empty().to_dict(),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "model.pth"
        torch.save(checkpoint, checkpoint_path)
        loader.load_model(model, checkpoint_path)

    assert len(model.validate_calls) == 1
    assert model.resolve_device_calls == ["cpu"]
    assert model.applied_state is not None
    assert model.applied_state.idx_to_user_id == [1]
    assert model.applied_state.seen_items_by_user == {1: {10, 30}}
    assert model.build_tower_calls == [(1, 2)]
    assert torch.equal(model.loaded_state_dict["weight"], torch.tensor([3.0]))
    assert model.to_calls == [torch.device("cpu")]
    assert model.invalidate_calls == 1
    assert model.eval_calls == 1


def test_load_model_raises_when_checkpoint_is_missing():
    loader = TwoTowerModelLoader()
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(FileNotFoundError):
            loader.load_model(StubLoadableModel(), Path(tmpdir) / "missing.pth")
