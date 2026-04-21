from __future__ import annotations

import math

import pandas as pd
import pytest
import torch
import torch.nn as nn

from twotower.src.backend.config import TwoTowerConfig
from twotower.fit import (
    EarlyStopping,
    FitInputs,
    NegativeSampling,
    PairwiseInteractionsDataset,
    TwoTowerTrainer,
    build_pairwise_loader,
)


class StubTrainableModel(nn.Module):
    def __init__(self, config: TwoTowerConfig):
        super().__init__()
        self.config = config
        self.user_id_to_idx = {1: 0, 2: 1}
        self.item_id_to_idx = {10: 0, 20: 1, 30: 2}
        self.user_embeddings: nn.Embedding | None = None
        self.item_embeddings: nn.Embedding | None = None
        self.build_tower_calls: list[tuple[int, int]] = []
        self.load_state_dict_calls = 0
        self.recall_at_k_calls: list[tuple[int]] = []

    def build_towers(self, num_users: int, num_items: int) -> None:
        self.build_tower_calls.append((num_users, num_items))
        self.user_embeddings = nn.Embedding(num_users, 4)
        self.item_embeddings = nn.Embedding(num_items, 4)

    def encode_users(self, user_input: torch.Tensor) -> torch.Tensor:
        if self.user_embeddings is None:
            raise RuntimeError("Embeddings are not initialized.")
        import torch.nn.functional as F
        return F.normalize(self.user_embeddings(user_input), dim=-1)

    def encode_items(self, item_input: torch.Tensor) -> torch.Tensor:
        if self.item_embeddings is None:
            raise RuntimeError("Embeddings are not initialized.")
        import torch.nn.functional as F
        return F.normalize(self.item_embeddings(item_input), dim=-1)

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        return (self.encode_users(user_input) * self.encode_items(item_input)).sum(dim=-1)

    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float:
        self.recall_at_k_calls.append((top_k,))
        return 0.5

    def load_state_dict(self, state_dict, strict: bool = True):
        self.load_state_dict_calls += 1
        return super().load_state_dict(state_dict, strict=strict)


@pytest.fixture
def interactions_data():
    positive_df = pd.DataFrame({"user_id": [1, 2], "banner_id": [10, 20], "label": [1.0, 1.0]})
    interactions_df = pd.DataFrame(
        {"user_id": [1, 1, 2, 2], "banner_id": [10, 20, 20, 10], "label": [1.0, 0.0, 1.0, 0.0]}
    )
    user_id_to_idx = {1: 0, 2: 1}
    item_id_to_idx = {10: 0, 20: 1, 30: 2}
    return positive_df, interactions_df, user_id_to_idx, item_id_to_idx


def test_pairwise_dataset_prefers_observed_negatives_when_ratio_is_one(interactions_data):
    positive_df, interactions_df, user_id_to_idx, item_id_to_idx = interactions_data
    dataset = PairwiseInteractionsDataset(
        positive_df=positive_df.iloc[[0]],
        interactions_df=interactions_df,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        num_items=3,
        observed_negative_sampling_ratio=1.0,
        seed=7,
    )

    user_idx, pos_item_idx, neg_item_idx = dataset[0]

    assert int(user_idx.item()) == 0
    assert int(pos_item_idx.item()) == 0
    assert int(neg_item_idx.item()) == 1


def test_pairwise_dataset_samples_only_non_positive_items_when_sampling_random_negatives(interactions_data):
    positive_df, interactions_df, user_id_to_idx, item_id_to_idx = interactions_data
    dataset = PairwiseInteractionsDataset(
        positive_df=positive_df.iloc[[0]],
        interactions_df=interactions_df[interactions_df["user_id"] == 1],
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        num_items=3,
        observed_negative_sampling_ratio=0.0,
        seed=11,
    )

    negative_item_indices = {int(dataset[0][2].item()) for _ in range(25)}

    assert negative_item_indices
    assert negative_item_indices.issubset({1, 2})
    assert 0 not in negative_item_indices


def test_build_pairwise_loader_returns_expected_training_triples(interactions_data):
    positive_df, interactions_df, user_id_to_idx, item_id_to_idx = interactions_data
    loader = build_pairwise_loader(
        positive_df=positive_df,
        interactions_df=interactions_df,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        num_items=3,
        batch_size=2,
        shuffle=False,
        observed_negative_sampling_ratio=1.0,
        seed=5,
    )

    user_batch, pos_item_batch, neg_item_batch = next(iter(loader))

    assert tuple(user_batch.shape) == (2,)
    assert tuple(pos_item_batch.shape) == (2,)
    assert tuple(neg_item_batch.shape) == (2,)
    assert torch.equal(user_batch, torch.tensor([0, 1]))
    assert torch.equal(pos_item_batch, torch.tensor([0, 1]))
    assert torch.equal(neg_item_batch, torch.tensor([1, 0]))


def test_trainer_fit_returns_history_and_restores_best_state(interactions_data):
    positive_df, interactions_df, _, _ = interactions_data
    config = TwoTowerConfig(epochs=2, batch_size=2, learning_rate=0.01, seed=13, device="cpu")
    model = StubTrainableModel(config)
    trainer = TwoTowerTrainer(config=config, device=torch.device("cpu"))
    fit_inputs = FitInputs(
        train_positive_df=positive_df,
        valid_positive_df=positive_df,
        train_interactions_df=interactions_df,
        valid_interactions_df=interactions_df,
        num_users=2,
        num_items=3,
    )

    fit_result = trainer.fit(model, fit_inputs, negative_sampling=NegativeSampling(observed_ratio=1.0))

    assert model.build_tower_calls == [(2, 3)]
    assert model.load_state_dict_calls == 1
    assert len(fit_result.history) == 2
    assert [r["epoch"] for r in fit_result.history] == [1.0, 2.0]
    for record in fit_result.history:
        assert "train_loss" in record
        assert "valid_loss" in record
        assert math.isfinite(record["train_loss"])
        assert math.isfinite(record["valid_loss"])


def test_trainer_fit_with_in_batch_loss_produces_finite_loss(interactions_data):
    positive_df, interactions_df, _, _ = interactions_data
    config = TwoTowerConfig(
        epochs=2, batch_size=2, learning_rate=0.01, seed=13, device="cpu",
        eval_during_training=False,
    )
    model = StubTrainableModel(config)
    trainer = TwoTowerTrainer(config=config, device=torch.device("cpu"))
    fit_inputs = FitInputs(
        train_positive_df=positive_df,
        valid_positive_df=positive_df,
        train_interactions_df=interactions_df,
        valid_interactions_df=interactions_df,
        num_users=2,
        num_items=3,
    )

    fit_result = trainer.fit(
        model, fit_inputs,
        negative_sampling=NegativeSampling(observed_ratio=1.0, in_batch_loss_weight=1.0),
        early_stopping=None,
    )

    assert len(fit_result.history) == 2
    for record in fit_result.history:
        assert math.isfinite(record["train_loss"])


def test_trainer_early_stopping_halts_before_max_epochs(interactions_data):
    positive_df, interactions_df, _, _ = interactions_data
    config = TwoTowerConfig(
        epochs=20, batch_size=2, learning_rate=0.0,  # lr=0 → loss never improves
        seed=13, device="cpu", eval_during_training=False,
    )
    model = StubTrainableModel(config)
    trainer = TwoTowerTrainer(config=config, device=torch.device("cpu"))
    fit_inputs = FitInputs(
        train_positive_df=positive_df,
        valid_positive_df=positive_df,
        train_interactions_df=interactions_df,
        valid_interactions_df=interactions_df,
        num_users=2,
        num_items=3,
    )

    fit_result = trainer.fit(model, fit_inputs, early_stopping=EarlyStopping(patience=2, min_delta=0.0))

    # first epoch sets best; epochs 2 and 3 show no improvement → stop at epoch 3
    assert len(fit_result.history) == 3


def test_trainer_early_stopping_on_recall_metric(interactions_data):
    positive_df, interactions_df, _, _ = interactions_data
    config = TwoTowerConfig(
        epochs=10, batch_size=2, learning_rate=0.0,
        seed=13, device="cpu", eval_during_training=False, eval_top_ks=(10,),
    )
    model = StubTrainableModel(config)
    trainer = TwoTowerTrainer(config=config, device=torch.device("cpu"))
    fit_inputs = FitInputs(
        train_positive_df=positive_df,
        valid_positive_df=positive_df,
        train_interactions_df=interactions_df,
        valid_interactions_df=interactions_df,
        num_users=2,
        num_items=3,
    )

    # recall_at_k stub always returns 0.5 → no improvement → stops after patience+1 epochs
    fit_result = trainer.fit(
        model, fit_inputs,
        early_stopping=EarlyStopping(patience=2, metric="recall_at_10", min_delta=0.0),
    )

    assert len(fit_result.history) == 3
    # recall must be in history even though eval_during_training=False
    for record in fit_result.history:
        assert "recall_at_10" in record


def test_trainer_no_early_stopping_runs_all_epochs(interactions_data):
    positive_df, interactions_df, _, _ = interactions_data
    config = TwoTowerConfig(
        epochs=3, batch_size=2, learning_rate=0.0,
        seed=13, device="cpu", eval_during_training=False,
    )
    model = StubTrainableModel(config)
    trainer = TwoTowerTrainer(config=config, device=torch.device("cpu"))
    fit_inputs = FitInputs(
        train_positive_df=positive_df,
        valid_positive_df=positive_df,
        train_interactions_df=interactions_df,
        valid_interactions_df=interactions_df,
        num_users=2,
        num_items=3,
    )

    fit_result = trainer.fit(model, fit_inputs, early_stopping=None)

    assert len(fit_result.history) == 3


def test_trainer_fit_computes_recall_metrics_when_eval_during_training(interactions_data):
    positive_df, interactions_df, _, _ = interactions_data
    config = TwoTowerConfig(
        epochs=2, batch_size=2, learning_rate=0.01,
        seed=13, device="cpu", eval_during_training=True, eval_top_ks=(10, 50),
    )
    model = StubTrainableModel(config)
    trainer = TwoTowerTrainer(config=config, device=torch.device("cpu"))
    fit_inputs = FitInputs(
        train_positive_df=positive_df,
        valid_positive_df=positive_df,
        train_interactions_df=interactions_df,
        valid_interactions_df=interactions_df,
        num_users=2,
        num_items=3,
    )

    fit_result = trainer.fit(model, fit_inputs)

    # recall_at_k called once per top-k per epoch: 2 top-ks * 2 epochs
    assert len(model.recall_at_k_calls) == 4
    assert sorted({call[0] for call in model.recall_at_k_calls}) == [10, 50]
    for record in fit_result.history:
        assert record["recall_at_10"] == pytest.approx(0.5)
        assert record["recall_at_50"] == pytest.approx(0.5)
