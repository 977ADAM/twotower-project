from __future__ import annotations

import pandas as pd
import pytest

from twotower.src.backend.config import TwoTowerConfig
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

    def make_loader(self, *, positive_df, interactions_df, shuffle) -> object:
        self.make_loader_calls.append({"positive_df": positive_df, "interactions_df": interactions_df, "shuffle": shuffle})
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


@pytest.fixture
def evaluator_setup():
    test_input_df = pd.DataFrame({
        "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "user_id": [1, 2, 3],
        "banner_id": [10, 20, 30],
        "label": [1.0, 0.0, 1.0],
    })
    prepared_test_df = test_input_df.iloc[[0, 1]].copy()
    positive_test_df = prepared_test_df.iloc[[0]].copy()
    evaluate_inputs = EvaluateInputs(
        test_input_df=test_input_df,
        prepared_test_df=prepared_test_df,
        positive_test_df=positive_test_df,
        input_row_count=3,
        unknown_user_row_count=1,
        unknown_item_row_count=0,
    )
    return test_input_df, evaluate_inputs, TwoTowerEvaluator()


def test_evaluate_aggregates_metrics_and_uses_default_top_k(evaluator_setup):
    test_input_df, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)

    metrics = evaluator.evaluate(model, test_input_df)

    assert model.ensure_fitted_calls == 1
    assert len(model.make_loader_calls) == 1
    assert not model.make_loader_calls[0]["shuffle"]
    assert model.evaluate_loader_calls == [({"loader": "ok"}, "test")]
    assert model.recall_calls == [5, 10]
    assert model.popularity_recall_calls == [5, 10]
    assert metrics["test_loss"] == 0.25
    assert metrics["recall_at_5"] == 0.4
    assert metrics["recall_at_10"] == 0.6
    assert metrics["recall_at_k"] == 0.6
    assert metrics["popularity_recall_at_k"] == 0.3
    assert metrics["test_input_rows"] == 3.0
    assert metrics["test_rows_used"] == 2.0
    assert metrics["test_rows_filtered"] == 1.0
    assert metrics["test_unknown_user_rows"] == 1.0
    assert metrics["test_unknown_item_rows"] == 0.0
    assert metrics["test_positive_pairs_used_for_loss"] == 1.0
    assert metrics["test_eval_user_count"] == 2.0


def test_evaluate_respects_top_k_override(evaluator_setup):
    test_input_df, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)

    metrics = evaluator.evaluate(model, test_input_df, top_k=3)

    assert model.recall_calls == [5, 3]
    assert model.popularity_recall_calls == [5, 3]
    assert metrics["recall_at_3"] == 0.2
    assert metrics["recall_at_k"] == 0.2
    assert metrics["popularity_recall_at_3"] == 0.05
    assert metrics["popularity_recall_at_k"] == 0.05


def test_evaluate_raises_for_empty_prepared_test_set(evaluator_setup):
    test_input_df, evaluate_inputs, evaluator = evaluator_setup
    empty_model = StubEvaluableModel(EvaluateInputs(
        test_input_df=test_input_df.iloc[:0].copy(),
        prepared_test_df=evaluate_inputs.prepared_test_df.iloc[:0].copy(),
        positive_test_df=evaluate_inputs.positive_test_df.iloc[:0].copy(),
        input_row_count=0,
        unknown_user_row_count=0,
        unknown_item_row_count=0,
    ))

    with pytest.raises(RuntimeError, match="Evaluation dataset is empty"):
        evaluator.evaluate(empty_model, test_input_df)
