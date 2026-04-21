from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd
from torch.utils.data import DataLoader

from twotower.src.backend.config import TwoTowerConfig


@dataclass(slots=True)
class EvaluateInputs:
    """Prepared test artifacts required by the evaluation service."""

    test_input_df: pd.DataFrame
    prepared_test_df: pd.DataFrame
    positive_test_df: pd.DataFrame
    input_row_count: int
    unknown_user_row_count: int
    unknown_item_row_count: int


class EvaluableTwoTower(Protocol):
    """Minimal model contract required by the evaluation module."""

    config: TwoTowerConfig

    def ensure_fitted(self) -> None:
        ...

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs:
        ...

    def make_loader(
        self,
        *,
        positive_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader:
        ...

    def evaluate_loader(self, loader: DataLoader, prefix: str = "valid") -> dict[str, float]:
        ...

    def resolve_eval_top_ks(self, top_k: int | None) -> list[int]:
        ...

    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int) -> float:
        ...

    def popularity_recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int) -> float:
        ...

    def get_eval_user_ids(self, evaluation_df: pd.DataFrame) -> list[int]:
        ...


class TwoTowerEvaluator:
    """Evaluate a two-tower model through a minimal protocol interface."""

    def evaluate(
        self,
        model: EvaluableTwoTower,
        X_test: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, float]:
        model.ensure_fitted()
        evaluate_inputs = model.build_evaluate_inputs(X_test)
        if evaluate_inputs.prepared_test_df.empty:
            raise RuntimeError(
                "Evaluation dataset is empty after filtering out unknown users and items."
            )

        metrics = model.evaluate_loader(
            model.make_loader(
                positive_df=evaluate_inputs.positive_test_df,
                interactions_df=evaluate_inputs.prepared_test_df,
                shuffle=False,
            ),
            prefix="test",
        )
        for eval_top_k in model.resolve_eval_top_ks(top_k):
            metrics[f"recall_at_{eval_top_k}"] = model.recall_at_k(
                evaluate_inputs.prepared_test_df,
                eval_top_k,
            )
            metrics[f"popularity_recall_at_{eval_top_k}"] = model.popularity_recall_at_k(
                evaluate_inputs.prepared_test_df,
                eval_top_k,
            )

        selected_top_k = model.config.top_k if top_k is None else int(top_k)
        metrics["recall_at_k"] = metrics[f"recall_at_{selected_top_k}"]
        metrics["popularity_recall_at_k"] = metrics[f"popularity_recall_at_{selected_top_k}"]
        metrics["test_input_rows"] = float(evaluate_inputs.input_row_count)
        metrics["test_rows_used"] = float(len(evaluate_inputs.prepared_test_df))
        metrics["test_rows_filtered"] = float(
            evaluate_inputs.input_row_count - len(evaluate_inputs.prepared_test_df)
        )
        metrics["test_unknown_user_rows"] = float(evaluate_inputs.unknown_user_row_count)
        metrics["test_unknown_item_rows"] = float(evaluate_inputs.unknown_item_row_count)
        metrics["test_positive_rate"] = float(evaluate_inputs.prepared_test_df["label"].mean())
        metrics["test_positive_pairs_used_for_loss"] = float(len(evaluate_inputs.positive_test_df))
        metrics["test_eval_user_count"] = float(
            len(model.get_eval_user_ids(evaluate_inputs.prepared_test_df))
        )
        metrics["test_rows_filtered_ratio"] = (
            float(evaluate_inputs.input_row_count - len(evaluate_inputs.prepared_test_df))
            / max(float(evaluate_inputs.input_row_count), 1.0)
        )
        return metrics
