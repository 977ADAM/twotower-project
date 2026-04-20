from __future__ import annotations

import math
import unittest

import pandas as pd
import torch
import torch.nn as nn

from twotower.config import TwoTowerConfig
from twotower.fit import (
    FitInputs,
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

    def build_towers(self, num_users: int, num_items: int) -> None:
        self.build_tower_calls.append((num_users, num_items))
        self.user_embeddings = nn.Embedding(num_users, 4)
        self.item_embeddings = nn.Embedding(num_items, 4)

    def score_pairs(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
    ) -> torch.Tensor:
        if self.user_embeddings is None or self.item_embeddings is None:
            raise RuntimeError("Embeddings are not initialized.")

        user_vectors = self.user_embeddings(user_input)
        item_vectors = self.item_embeddings(item_input)
        return (user_vectors * item_vectors).sum(dim=-1)

    def retrieval_logits(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
    ) -> torch.Tensor:
        if self.user_embeddings is None or self.item_embeddings is None:
            raise RuntimeError("Embeddings are not initialized.")

        user_vectors = self.user_embeddings(user_input)
        item_vectors = self.item_embeddings(item_input)
        return torch.matmul(user_vectors, item_vectors.T)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.load_state_dict_calls += 1
        return super().load_state_dict(state_dict, strict=strict)


class FitModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.positive_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "banner_id": [10, 20],
                "label": [1.0, 1.0],
            }
        )
        self.interactions_df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2],
                "banner_id": [10, 20, 20, 10],
                "label": [1.0, 0.0, 1.0, 0.0],
            }
        )
        self.user_id_to_idx = {1: 0, 2: 1}
        self.item_id_to_idx = {10: 0, 20: 1, 30: 2}

    def test_pairwise_dataset_prefers_observed_negatives_when_ratio_is_one(self):
        dataset = PairwiseInteractionsDataset(
            positive_df=self.positive_df.iloc[[0]],
            interactions_df=self.interactions_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            num_items=3,
            observed_negative_sampling_ratio=1.0,
            seed=7,
        )

        user_idx, pos_item_idx, neg_item_idx = dataset[0]

        self.assertEqual(int(user_idx.item()), 0)
        self.assertEqual(int(pos_item_idx.item()), 0)
        self.assertEqual(int(neg_item_idx.item()), 1)

    def test_pairwise_dataset_samples_only_non_positive_items_when_sampling_random_negatives(self):
        dataset = PairwiseInteractionsDataset(
            positive_df=self.positive_df.iloc[[0]],
            interactions_df=self.interactions_df[self.interactions_df["user_id"] == 1],
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            num_items=3,
            observed_negative_sampling_ratio=0.0,
            seed=11,
        )

        negative_item_indices = {int(dataset[0][2].item()) for _ in range(25)}

        self.assertTrue(negative_item_indices)
        self.assertTrue(negative_item_indices.issubset({1, 2}))
        self.assertNotIn(0, negative_item_indices)

    def test_build_pairwise_loader_returns_expected_training_triples(self):
        loader = build_pairwise_loader(
            positive_df=self.positive_df,
            interactions_df=self.interactions_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            num_items=3,
            batch_size=2,
            shuffle=False,
            observed_negative_sampling_ratio=1.0,
            seed=5,
        )

        user_batch, pos_item_batch, neg_item_batch = next(iter(loader))

        self.assertEqual(tuple(user_batch.shape), (2,))
        self.assertEqual(tuple(pos_item_batch.shape), (2,))
        self.assertEqual(tuple(neg_item_batch.shape), (2,))
        self.assertTrue(torch.equal(user_batch, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(pos_item_batch, torch.tensor([0, 1])))
        self.assertTrue(torch.equal(neg_item_batch, torch.tensor([1, 0])))

    def test_trainer_fit_returns_history_and_restores_best_state(self):
        config = TwoTowerConfig(
            epochs=2,
            batch_size=2,
            learning_rate=0.01,
            observed_negative_sampling_ratio=1.0,
            seed=13,
            device="cpu",
        )
        model = StubTrainableModel(config)
        trainer = TwoTowerTrainer(config=config, device=torch.device("cpu"))
        fit_inputs = FitInputs(
            train_positive_df=self.positive_df,
            valid_positive_df=self.positive_df,
            train_interactions_df=self.interactions_df,
            valid_interactions_df=self.interactions_df,
            num_users=2,
            num_items=3,
        )

        fit_result = trainer.fit(model, fit_inputs)

        self.assertEqual(model.build_tower_calls, [(2, 3)])
        self.assertEqual(model.load_state_dict_calls, 1)
        self.assertEqual(len(fit_result.history), 2)
        self.assertEqual(
            [record["epoch"] for record in fit_result.history],
            [1.0, 2.0],
        )
        for record in fit_result.history:
            self.assertIn("train_loss", record)
            self.assertIn("valid_loss", record)
            self.assertTrue(math.isfinite(record["train_loss"]))
            self.assertTrue(math.isfinite(record["valid_loss"]))


if __name__ == "__main__":
    unittest.main()
