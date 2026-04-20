from __future__ import annotations

import unittest

import pandas as pd

from twotower.config import TwoTowerConfig
from twotower.evaluate import EvaluateInputs, TwoTowerEvaluator


class StubEvaluableModel:
    def __init__(self, evaluate_inputs: EvaluateInputs):
        self.config = TwoTowerConfig(top_k=10, eval_top_ks=(5, 10))
        self.evaluate_inputs = evaluate_inputs
        self.ensure_fitted_calls = 0
        self.make_loader_calls: list[dict[str, object]] = []
        self.evaluate_loader_calls: list[tuple[object, str]] = []
        self.recall_calls: list[int] = []
        self.popularity_recall_calls: list[int] = []
        self.last_input = None

    def ensure_fitted(self) -> None:
        self.ensure_fitted_calls += 1

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs:
        self.last_input = X_test
        return self.evaluate_inputs

    def make_loader(
        self,
        *,
        positive_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        shuffle: bool,
    ) -> object:
        self.make_loader_calls.append(
            {
                "positive_df": positive_df,
                "interactions_df": interactions_df,
                "shuffle": shuffle,
            }
        )
        return {"loader": "ok"}

    def evaluate_loader(self, loader: object, prefix: str = "valid") -> dict[str, float]:
        self.evaluate_loader_calls.append((loader, prefix))
        return {"test_loss": 0.25}

    def resolve_eval_top_ks(self, top_k: int | None) -> list[int]:
        return [5, 10] if top_k is None else [5, int(top_k)]

    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int) -> float:
        self.recall_calls.append(top_k)
        return {5: 0.4, 10: 0.6, 3: 0.2}[top_k]

    def popularity_recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int) -> float:
        self.popularity_recall_calls.append(top_k)
        return {5: 0.1, 10: 0.3, 3: 0.05}[top_k]

    def get_eval_user_ids(self, evaluation_df: pd.DataFrame) -> list[int]:
        return [1, 2]


class TwoTowerEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_input_df = pd.DataFrame(
            {
                "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
                "user_id": [1, 2, 3],
                "banner_id": [10, 20, 30],
                "label": [1.0, 0.0, 1.0],
            }
        )
        self.prepared_test_df = self.test_input_df.iloc[[0, 1]].copy()
        self.positive_test_df = self.prepared_test_df.iloc[[0]].copy()
        self.evaluate_inputs = EvaluateInputs(
            test_input_df=self.test_input_df,
            prepared_test_df=self.prepared_test_df,
            positive_test_df=self.positive_test_df,
            input_row_count=3,
            unknown_user_row_count=1,
            unknown_item_row_count=0,
        )
        self.model = StubEvaluableModel(self.evaluate_inputs)
        self.evaluator = TwoTowerEvaluator()

    def test_evaluate_aggregates_metrics_and_uses_default_top_k(self):
        metrics = self.evaluator.evaluate(self.model, self.test_input_df)

        self.assertEqual(self.model.ensure_fitted_calls, 1)
        self.assertEqual(len(self.model.make_loader_calls), 1)
        self.assertFalse(self.model.make_loader_calls[0]["shuffle"])
        self.assertEqual(self.model.evaluate_loader_calls, [({"loader": "ok"}, "test")])
        self.assertEqual(self.model.recall_calls, [5, 10])
        self.assertEqual(self.model.popularity_recall_calls, [5, 10])
        self.assertEqual(metrics["test_loss"], 0.25)
        self.assertEqual(metrics["recall_at_5"], 0.4)
        self.assertEqual(metrics["recall_at_10"], 0.6)
        self.assertEqual(metrics["recall_at_k"], 0.6)
        self.assertEqual(metrics["popularity_recall_at_k"], 0.3)
        self.assertEqual(metrics["test_input_rows"], 3.0)
        self.assertEqual(metrics["test_rows_used"], 2.0)
        self.assertEqual(metrics["test_rows_filtered"], 1.0)
        self.assertEqual(metrics["test_unknown_user_rows"], 1.0)
        self.assertEqual(metrics["test_unknown_item_rows"], 0.0)
        self.assertEqual(metrics["test_positive_pairs_used_for_loss"], 1.0)
        self.assertEqual(metrics["test_eval_user_count"], 2.0)

    def test_evaluate_respects_top_k_override(self):
        metrics = self.evaluator.evaluate(self.model, self.test_input_df, top_k=3)

        self.assertEqual(self.model.recall_calls, [5, 3])
        self.assertEqual(self.model.popularity_recall_calls, [5, 3])
        self.assertEqual(metrics["recall_at_3"], 0.2)
        self.assertEqual(metrics["recall_at_k"], 0.2)
        self.assertEqual(metrics["popularity_recall_at_3"], 0.05)
        self.assertEqual(metrics["popularity_recall_at_k"], 0.05)

    def test_evaluate_raises_for_empty_prepared_test_set(self):
        empty_model = StubEvaluableModel(
            EvaluateInputs(
                test_input_df=self.test_input_df.iloc[:0].copy(),
                prepared_test_df=self.prepared_test_df.iloc[:0].copy(),
                positive_test_df=self.positive_test_df.iloc[:0].copy(),
                input_row_count=0,
                unknown_user_row_count=0,
                unknown_item_row_count=0,
            )
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "Evaluation dataset is empty after filtering out unknown users and items.",
        ):
            self.evaluator.evaluate(empty_model, self.test_input_df)


if __name__ == "__main__":
    unittest.main()
