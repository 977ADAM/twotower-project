from __future__ import annotations

import pandas as pd
import pytest
import torch

from twotower._src.config import _Config
from twotower._src.retrieval.evaluate import EvaluateInputs, TwoTowerEvaluator


class StubEvaluableModel:
    """Stub satisfying the updated _Evaluable protocol (no evaluate_loader/recall_at_k on model)."""

    def __init__(self, evaluate_inputs: EvaluateInputs):
        self.config = _Config(top_k=10, eval_top_ks=(5, 10))
        self.device = torch.device("cpu")
        self.evaluate_inputs = evaluate_inputs
        self.query_id_to_idx = {1: 0}
        self.candidate_id_to_idx = {10: 0, 20: 1, 30: 2}
        self.idx_to_candidate_id = [10, 20, 30]
        self.ensure_fitted_calls = 0
        self.make_loader_calls: list[dict[str, object]] = []
        self.eval_calls = 0
        # user idx 0 → [1, 0], item indices 0,1,2 → [1,0],[0,1],[0,1]
        self._user_embeddings_by_idx = {0: torch.tensor([1.0, 0.0])}
        self._item_embeddings_by_idx = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.0, 1.0]),
            2: torch.tensor([0.0, 1.0]),
        }

    def eval(self) -> object:
        self.eval_calls += 1
        return self

    def ensure_fitted(self) -> None:
        self.ensure_fitted_calls += 1

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs:
        return self.evaluate_inputs

    def make_loader(self, *, positive_df, interactions_df, shuffle) -> list:
        self.make_loader_calls.append(
            {"positive_df": positive_df, "interactions_df": interactions_df, "shuffle": shuffle}
        )
        return []  # empty loader → evaluate_loader yields test_loss=0.0

    def resolve_eval_top_ks(self, top_k: int | None) -> list[int]:
        return [5, 10] if top_k is None else [5, int(top_k)]

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        return torch.zeros(user_input.size(0))

    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._user_embeddings_by_idx[int(idx)] for idx in user_input])

    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._item_embeddings_by_idx[int(idx)] for idx in item_input])

    def get_seen_candidates_by_query(self) -> dict[int, set[int]]:
        return {}

    def get_train_positive_item_ranking(self) -> list[int]:
        return []


@pytest.fixture
def evaluator_setup():
    test_input_df = pd.DataFrame({
        "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "query_id": [1, 2, 3],
        "candidate_id": [10, 20, 30],
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
    # evaluate_loader, recall_at_k, popularity_recall_at_k now live on TwoTowerEvaluator
    assert metrics["test_loss"] == 0.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["popularity_recall_at_5"] == 0.0
    assert metrics["popularity_recall_at_10"] == 0.0
    assert metrics["popularity_recall_at_k"] == 0.0
    assert metrics["test_input_rows"] == 3.0
    assert metrics["test_rows_used"] == 2.0
    assert metrics["test_rows_filtered"] == 1.0
    assert metrics["test_unknown_user_rows"] == 1.0
    assert metrics["test_unknown_item_rows"] == 0.0
    assert metrics["test_positive_pairs_used_for_loss"] == 1.0
    assert metrics["test_eval_user_count"] == 1.0


def test_evaluate_respects_top_k_override(evaluator_setup):
    test_input_df, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)

    metrics = evaluator.evaluate(model, test_input_df, top_k=3)

    assert metrics["recall_at_3"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["popularity_recall_at_3"] == 0.0
    assert metrics["popularity_recall_at_k"] == 0.0


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


# ── Tests for the methods that moved from TwoTower to TwoTowerEvaluator ──────


def test_get_eval_user_ids_returns_users_with_positive_labels(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    df = pd.DataFrame({
        "query_id": [1, 2, 3, 1],
        "candidate_id": [10, 20, 30, 20],
        "label": [1.0, 0.0, 1.0, 1.0],
    })

    user_ids = evaluator.get_eval_user_ids(model, df)

    assert set(user_ids) == {1, 3}


def test_get_eval_user_ids_returns_empty_for_no_positives(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    df = pd.DataFrame({"query_id": [1, 2], "candidate_id": [10, 20], "label": [0.0, 0.0]})

    assert evaluator.get_eval_user_ids(model, df) == []


def test_evaluate_loader_returns_zero_loss_for_empty_loader(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)

    metrics = evaluator.evaluate_loader(model, [], prefix="valid")

    assert metrics == {"valid_loss": 0.0}


def test_recall_at_k_returns_one_for_item_at_top(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    # prepared_test_df has query_id=1 with label=1.0 and candidate_id=10
    # model.encode_queries([0]) → [[1,0]], encode_candidates([0,1,2]) → [[1,0],[0,1],[0,1]]
    # user 1 scores: [1.0, 0.0, 0.0] → item 10 is top-1 → recall = 1.0
    recall = evaluator.recall_at_k(model, evaluate_inputs.prepared_test_df, top_k=1)

    assert recall == 1.0


def test_recall_at_k_returns_zero_for_no_positive_users(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    df = pd.DataFrame({"query_id": [1], "candidate_id": [10], "label": [0.0]})

    assert evaluator.recall_at_k(model, df, top_k=5) == 0.0


def test_popularity_recall_at_k_returns_zero_for_empty_ranking(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    # model.get_train_positive_item_ranking() → [] → popularity_recall = 0.0

    assert evaluator.popularity_recall_at_k(model, evaluate_inputs.prepared_test_df, top_k=5) == 0.0


def test_popularity_recall_at_k_returns_correct_score_for_non_empty_ranking(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup

    class ModelWithRanking(StubEvaluableModel):
        def get_train_positive_item_ranking(self) -> list[int]:
            return [10, 20, 30]  # item 10 ranked first

    model = ModelWithRanking(evaluate_inputs)
    # prepared_test_df: query_id=1 has label=1.0, candidate_id=10
    # popularity top-1 is item 10, no seen candidates → predicted = {10}
    # actual_items = {10} → recall = 1.0
    recall = evaluator.popularity_recall_at_k(model, evaluate_inputs.prepared_test_df, top_k=1)

    assert recall == 1.0
