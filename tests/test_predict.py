from __future__ import annotations

import unittest

import torch

from twotower.config import TwoTowerConfig
from twotower.predict import TwoTowerPredictor


class StubPredictableModel:
    def __init__(self):
        self.config = TwoTowerConfig(top_k=2)
        self.user_id_to_idx = {1: 0, 2: 1}
        self.item_id_to_idx = {10: 0, 20: 1, 30: 2}
        self.idx_to_user_id = [1, 2]
        self.idx_to_item_id = [10, 20, 30]
        self.eval_calls = 0

        self._user_embeddings = {
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
        }
        self._item_embeddings = {
            10: torch.tensor([1.0, 0.0]),
            20: torch.tensor([0.8, 0.2]),
            30: torch.tensor([0.0, 1.0]),
        }
        self._seen_items_by_user = {
            1: {10},
            2: set(),
        }

    def eval(self):
        self.eval_calls += 1

    def get_candidate_item_embeddings(
        self,
        item_ids: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        return torch.stack([self._item_embeddings[item_id] for item_id in item_ids]), list(item_ids)

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        return self._user_embeddings[user_id]

    def get_seen_items_by_user(self) -> dict[int, set[int]]:
        return self._seen_items_by_user


class TwoTowerPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = StubPredictableModel()
        self.predictor = TwoTowerPredictor()

    def test_predict_excludes_seen_items_by_default(self):
        predictions = self.predictor.predict(self.model, user_ids=[1], top_k=2)

        self.assertEqual(self.model.eval_calls, 1)
        self.assertEqual(
            [row["banner_id"] for row in predictions[1]],
            [20, 30],
        )

    def test_predict_deduplicates_ids_and_skips_unknown_ids_by_default(self):
        predictions = self.predictor.predict(
            self.model,
            user_ids=[999, 1, 1],
            item_ids=[20, 20, 30, 999],
            top_k=5,
        )

        self.assertEqual(list(predictions.keys()), [1])
        self.assertEqual(
            [row["banner_id"] for row in predictions[1]],
            [20, 30],
        )

    def test_predict_strict_raises_for_unknown_ids(self):
        with self.assertRaisesRegex(ValueError, "unknown user_ids: \\[999\\]"):
            self.predictor.predict(self.model, user_ids=[999], strict=True)

    def test_predict_top_k_item_ids_for_user_supports_recall_style_usage(self):
        item_embeddings, item_ids = self.model.get_candidate_item_embeddings(self.model.idx_to_item_id)

        predicted_item_ids = self.predictor.predict_top_k_item_ids_for_user(
            self.model,
            user_id=1,
            item_embeddings=item_embeddings,
            item_ids=item_ids,
            top_k=2,
            excluded_item_ids={10},
        )

        self.assertEqual(predicted_item_ids, {20, 30})


if __name__ == "__main__":
    unittest.main()
