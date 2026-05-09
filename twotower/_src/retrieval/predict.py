from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd
import torch

from twotower._src.protocols import _HasConfig, _HasIDMappings


class _Predictable(_HasIDMappings, _HasConfig, Protocol):
    """Minimal model contract required by the prediction module."""

    device: torch.device
    query_col: str
    candidate_col: str

    def eval(self) -> object: ...
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...
    def get_seen_candidates_by_query(self) -> dict[int, set[int]]: ...


class TwoTowerPredictor:
    """Generate top-k recommendations from a minimal prediction interface."""

    def __init__(self) -> None:
        self._cached_all_item_embeddings: torch.Tensor | None = None
        self._cached_all_item_ids: list[int] | None = None

    def predict(
        self,
        model: _Predictable,
        *,
        user_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        exclude_seen: bool = True,
        strict: bool = False,
    ) -> pd.DataFrame:
        resolved_user_ids, candidate_item_ids, resolved_top_k = self.prepare_prediction_inputs(
            model,
            user_ids=user_ids,
            item_ids=item_ids,
            top_k=top_k,
            strict=strict,
        )

        empty = pd.DataFrame(columns=[model.query_col, model.candidate_col, "score", "rank"])
        if not resolved_user_ids or not candidate_item_ids:
            return empty

        model.eval()
        item_embeddings, candidate_item_ids = self.get_candidate_item_embeddings(model, candidate_item_ids)
        seen_candidates_by_query = model.get_seen_candidates_by_query() if exclude_seen else {}

        rows: list[dict[str, object]] = []
        for query_id in resolved_user_ids:
            scored_items = self.score_top_k_for_user(
                model,
                query_id=query_id,
                item_embeddings=item_embeddings,
                item_ids=candidate_item_ids,
                top_k=resolved_top_k,
                excluded_item_ids=seen_candidates_by_query.get(query_id, set()),
            )
            for rank, (candidate_id, score) in enumerate(scored_items, start=1):
                rows.append({model.query_col: query_id, model.candidate_col: candidate_id, "score": score, "rank": rank})

        return pd.DataFrame(rows, columns=[model.query_col, model.candidate_col, "score", "rank"])

    def prepare_prediction_inputs(
        self,
        model: _Predictable,
        *,
        user_ids: Sequence[int] | None,
        item_ids: Sequence[int] | None,
        top_k: int | None,
        strict: bool = False,
    ) -> tuple[list[int], list[int], int]:
        """Validate and normalize prediction inputs."""
        resolved_top_k = model.config.top_k if top_k is None else int(top_k)
        if resolved_top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")

        resolved_user_ids = (
            self._deduplicate_ids(user_ids)
            if user_ids is not None
            else model.idx_to_query_id[: min(10, len(model.idx_to_query_id))]
        )
        resolved_item_ids = (
            self._deduplicate_ids(item_ids)
            if item_ids is not None
            else list(model.idx_to_candidate_id)
        )

        unknown_user_ids = [
            query_id for query_id in resolved_user_ids if query_id not in model.query_id_to_idx
        ]
        unknown_item_ids = [
            candidate_id for candidate_id in resolved_item_ids if candidate_id not in model.candidate_id_to_idx
        ]
        if strict and (unknown_user_ids or unknown_item_ids):
            error_messages: list[str] = []
            if unknown_user_ids:
                error_messages.append(f"unknown user_ids: {unknown_user_ids[:5]}")
            if unknown_item_ids:
                error_messages.append(f"unknown item_ids: {unknown_item_ids[:5]}")
            raise ValueError("Prediction received " + "; ".join(error_messages) + ".")

        available_user_ids = [
            int(query_id) for query_id in resolved_user_ids if int(query_id) in model.query_id_to_idx
        ]
        available_item_ids = [
            int(candidate_id) for candidate_id in resolved_item_ids if int(candidate_id) in model.candidate_id_to_idx
        ]
        return available_user_ids, available_item_ids, resolved_top_k

    def score_top_k_for_user(
        self,
        model: _Predictable,
        *,
        query_id: int,
        item_embeddings: torch.Tensor,
        item_ids: list[int],
        top_k: int,
        excluded_item_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if query_id not in model.query_id_to_idx:
            return []

        excluded_item_ids = excluded_item_ids or set()
        candidate_positions = [
            position for position, candidate_id in enumerate(item_ids)
            if candidate_id not in excluded_item_ids
        ]
        if not candidate_positions:
            return []

        user_embedding = self.get_user_embedding(model, query_id)
        candidate_embeddings = item_embeddings[candidate_positions]
        scores = torch.matmul(candidate_embeddings, user_embedding)

        k = min(top_k, scores.size(0))
        if k == 0:
            return []

        top_scores, top_positions = torch.topk(scores, k=k)
        return [
            (
                int(item_ids[candidate_positions[position]]),
                float(score),
            )
            for score, position in zip(top_scores.cpu().tolist(), top_positions.cpu().tolist())
        ]

    def predict_top_k_item_ids_for_user(
        self,
        model: _Predictable,
        *,
        query_id: int,
        item_embeddings: torch.Tensor,
        item_ids: list[int],
        top_k: int,
        excluded_item_ids: set[int] | None = None,
    ) -> set[int]:
        return {
            candidate_id
            for candidate_id, _score in self.score_top_k_for_user(
                model,
                query_id=query_id,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
                top_k=top_k,
                excluded_item_ids=excluded_item_ids,
            )
        }

    def get_candidate_item_embeddings(
        self,
        model: _Predictable,
        item_ids: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        """Return embeddings for `item_ids`, using the all-items cache for the full catalogue."""
        all_item_embeddings, all_item_ids = self._build_candidate_item_embeddings(model)
        if item_ids == all_item_ids:
            return all_item_embeddings, all_item_ids

        valid_pairs = [
            (model.candidate_id_to_idx[cid], cid)
            for cid in item_ids
            if cid in model.candidate_id_to_idx
        ]
        if not valid_pairs:
            return all_item_embeddings[:0], []

        positions, valid_ids = zip(*valid_pairs)
        return all_item_embeddings[list(positions)], list(valid_ids)

    def get_user_embedding(self, model: _Predictable, query_id: int) -> torch.Tensor:
        if query_id not in model.query_id_to_idx:
            raise KeyError(f"Unknown query_id: {query_id}")

        user_index = torch.tensor(
            [model.query_id_to_idx[query_id]],
            dtype=torch.long,
            device=model.device,
        )
        with torch.no_grad():
            return model.encode_queries(user_index).squeeze(0)

    def invalidate_cache(self) -> None:
        """Clear the cached item embeddings. Must be called when the model weights change."""
        self._cached_all_item_embeddings = None
        self._cached_all_item_ids = None

    def _build_candidate_item_embeddings(self, model: _Predictable) -> tuple[torch.Tensor, list[int]]:
        if self._cached_all_item_embeddings is None or self._cached_all_item_ids is None:
            item_ids = list(model.idx_to_candidate_id)
            item_indices = torch.tensor(
                [model.candidate_id_to_idx[candidate_id] for candidate_id in item_ids],
                dtype=torch.long,
                device=model.device,
            )
            with torch.no_grad():
                item_embeddings = model.encode_candidates(item_indices)
            self._cached_all_item_embeddings = item_embeddings
            self._cached_all_item_ids = item_ids
        return self._cached_all_item_embeddings, self._cached_all_item_ids

    @staticmethod
    def _deduplicate_ids(entity_ids: Sequence[int]) -> list[int]:
        deduplicated_ids: list[int] = []
        seen_ids: set[int] = set()
        for entity_id in entity_ids:
            normalized_id = int(entity_id)
            if normalized_id in seen_ids:
                continue
            seen_ids.add(normalized_id)
            deduplicated_ids.append(normalized_id)
        return deduplicated_ids
