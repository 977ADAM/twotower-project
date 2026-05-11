from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from twotower._src.metrics import mean_recall, user_recall, MetricInputs, MetricResult
from twotower._src.protocols import _HasConfig, _HasIDMappings


@dataclass(slots=True)
class EvaluateInputs:
    """Prepared test artifacts required by the evaluation service."""

    test_input_df: pd.DataFrame
    prepared_test_df: pd.DataFrame
    positive_test_df: pd.DataFrame
    input_row_count: int
    unknown_user_row_count: int
    unknown_item_row_count: int


class _Evaluable(_HasIDMappings, _HasConfig, Protocol):
    """Minimal model contract required by the evaluation module."""

    device: torch.device

    def eval(self) -> object: ...
    def ensure_fitted(self) -> None: ...

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs: ...

    def make_loader(
        self,
        *,
        positive_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader[Any]: ...

    def resolve_eval_top_ks(self, top_k: int | None) -> list[int]: ...

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor: ...
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...

    def get_seen_candidates_by_query(self) -> dict[int, set[int]]: ...
    def get_train_positive_item_ranking(self) -> list[int]: ...


class TwoTowerEvaluator:
    """Evaluate a two-tower model through a minimal protocol interface."""

    def evaluate(
        self,
        model: _Evaluable,
        X_test: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, float]:
        model.ensure_fitted()
        evaluate_inputs = model.build_evaluate_inputs(X_test)
        if evaluate_inputs.prepared_test_df.empty:
            raise RuntimeError(
                "Evaluation dataset is empty after filtering out unknown users and items."
            )

        metrics = self.evaluate_loader(
            model,
            model.make_loader(
                positive_df=evaluate_inputs.positive_test_df,
                interactions_df=evaluate_inputs.prepared_test_df,
                shuffle=False,
            ),
            prefix="test",
        )
        for eval_top_k in model.resolve_eval_top_ks(top_k):
            metrics[f"recall_at_{eval_top_k}"] = self.recall_at_k(
                model, evaluate_inputs.prepared_test_df, eval_top_k,
            )
            metrics[f"popularity_recall_at_{eval_top_k}"] = self.popularity_recall_at_k(
                model, evaluate_inputs.prepared_test_df, eval_top_k,
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
            len(self.get_eval_user_ids(model, evaluate_inputs.prepared_test_df))
        )
        metrics["test_rows_filtered_ratio"] = (
            float(evaluate_inputs.input_row_count - len(evaluate_inputs.prepared_test_df))
            / max(float(evaluate_inputs.input_row_count), 1.0)
        )
        return metrics

    def evaluate_loader(
        self,
        model: _Evaluable,
        loader: DataLoader[Any],
        prefix: str = "valid",
    ) -> dict[str, float]:
        model.eval()
        loss_sum = 0.0
        total = 0

        with torch.no_grad():
            for user_batch, pos_item_batch, neg_item_batch in loader:
                user_batch = user_batch.to(model.device)
                pos_item_batch = pos_item_batch.to(model.device)
                neg_item_batch = neg_item_batch.to(model.device)

                positive_scores = model.score_pairs(user_batch, pos_item_batch)
                negative_scores = model.score_pairs(user_batch, neg_item_batch)
                loss: torch.Tensor = -F.logsigmoid(positive_scores - negative_scores).mean()

                batch_size = user_batch.size(0)
                loss_sum += loss.item() * batch_size
                total += batch_size

        return {f"{prefix}_loss": loss_sum / max(total, 1)}

    def get_eval_user_ids(self, model: _Evaluable, evaluation_df: pd.DataFrame) -> list[int]:
        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        if positive_df.empty:
            return []
        return list(
            positive_df["query_id"]
            .drop_duplicates()
            .head(model.config.max_eval_users)
            .astype(int)
        )

    def recall_at_k(
        self,
        model: _Evaluable,
        evaluation_df: pd.DataFrame,
        top_k: int,
        exclude_seen: bool = True,
        item_embeddings: torch.Tensor | None = None,
    ) -> float:
        candidate_user_ids = self.get_eval_user_ids(model, evaluation_df)
        if not candidate_user_ids:
            return 0.0

        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        seen_candidates_by_query = model.get_seen_candidates_by_query() if exclude_seen else {}

        item_ids = list(model.idx_to_candidate_id)
        if item_embeddings is None:
            item_indices = torch.tensor(
                [model.candidate_id_to_idx[cid] for cid in item_ids],
                dtype=torch.long,
                device=model.device,
            )
            with torch.no_grad():
                item_embeddings = model.encode_candidates(item_indices)

        recalls = []
        for query_id in candidate_user_ids:
            actual_items = set(
                positive_df.loc[positive_df["query_id"] == query_id, "candidate_id"].astype(int)
            )
            if not actual_items or query_id not in model.query_id_to_idx:
                continue

            excluded = seen_candidates_by_query.get(query_id, set())
            candidate_positions = [
                pos for pos, cid in enumerate(item_ids) if cid not in excluded
            ]
            if not candidate_positions:
                recalls.append(MetricResult(value=0.0))
                continue

            user_idx = torch.tensor(
                [model.query_id_to_idx[query_id]], dtype=torch.long, device=model.device,
            )
            with torch.no_grad():
                user_embedding = model.encode_queries(user_idx).squeeze(0)

            candidate_embeddings = item_embeddings[candidate_positions]
            scores = torch.matmul(candidate_embeddings, user_embedding)
            k = min(top_k, scores.size(0))
            _, top_positions = torch.topk(scores, k=k)
            predicted_items = {item_ids[candidate_positions[p]] for p in top_positions.cpu().tolist()}
            recalls.append(user_recall(MetricInputs(actual=frozenset(actual_items), predicted=frozenset(predicted_items))))

        return mean_recall(recalls).value

    def popularity_recall_at_k(
        self,
        model: _Evaluable,
        evaluation_df: pd.DataFrame,
        top_k: int,
    ) -> float:
        candidate_user_ids = self.get_eval_user_ids(model, evaluation_df)
        if not candidate_user_ids:
            return 0.0

        popularity_ranking = model.get_train_positive_item_ranking()
        if not popularity_ranking:
            return 0.0

        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        seen_candidates_by_query = model.get_seen_candidates_by_query()
        recalls = []
        for query_id in candidate_user_ids:
            actual_items = set(
                positive_df.loc[positive_df["query_id"] == query_id, "candidate_id"].astype(int)
            )
            if not actual_items:
                continue

            excluded_item_ids = seen_candidates_by_query.get(query_id, set())
            predicted_items: list[int] = []
            for candidate_id in popularity_ranking:
                if candidate_id in excluded_item_ids:
                    continue
                predicted_items.append(candidate_id)
                if len(predicted_items) == top_k:
                    break

            recalls.append(user_recall(MetricInputs(actual=frozenset(actual_items), predicted=frozenset(predicted_items))))

        return mean_recall(recalls).value
