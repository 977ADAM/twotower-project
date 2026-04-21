from __future__ import annotations

from typing import Protocol, Sequence

import torch
import torch.nn.functional as F

from twotower.src.backend.config import TwoTowerConfig


class PredictableTwoTower(Protocol):
    """Minimal model contract required by the prediction module."""

    config: TwoTowerConfig
    user_id_to_idx: dict[int, int]
    item_id_to_idx: dict[int, int]
    idx_to_user_id: list[int]
    idx_to_item_id: list[int]

    def eval(self):
        ...

    def get_candidate_item_embeddings(
        self,
        item_ids: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        ...

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        ...

    def get_seen_items_by_user(self) -> dict[int, set[int]]:
        ...


class TwoTowerPredictor:
    """Generate top-k recommendations from a minimal prediction interface."""

    def predict(
        self,
        model: PredictableTwoTower,
        *,
        user_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        exclude_seen: bool = True,
        strict: bool = False,
    ) -> dict[int, list[dict[str, float]]]:
        resolved_user_ids, candidate_item_ids, resolved_top_k = self.prepare_prediction_inputs(
            model,
            user_ids=user_ids,
            item_ids=item_ids,
            top_k=top_k,
            strict=strict,
        )
        if not resolved_user_ids or not candidate_item_ids:
            return {}

        model.eval()
        item_embeddings, candidate_item_ids = model.get_candidate_item_embeddings(candidate_item_ids)
        seen_items_by_user = model.get_seen_items_by_user() if exclude_seen else {}

        predictions: dict[int, list[dict[str, float]]] = {}
        for user_id in resolved_user_ids:
            scored_items = self.score_top_k_for_user(
                model,
                user_id=user_id,
                item_embeddings=item_embeddings,
                item_ids=candidate_item_ids,
                top_k=resolved_top_k,
                excluded_item_ids=seen_items_by_user.get(user_id, set()),
            )
            predictions[user_id] = [
                {
                    "banner_id": item_id,
                    "score": score,
                }
                for item_id, score in scored_items
            ]

        return predictions

    def prepare_prediction_inputs(
        self,
        model: PredictableTwoTower,
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
            else model.idx_to_user_id[: min(10, len(model.idx_to_user_id))]
        )
        resolved_item_ids = (
            self._deduplicate_ids(item_ids)
            if item_ids is not None
            else list(model.idx_to_item_id)
        )

        unknown_user_ids = [
            user_id for user_id in resolved_user_ids if user_id not in model.user_id_to_idx
        ]
        unknown_item_ids = [
            item_id for item_id in resolved_item_ids if item_id not in model.item_id_to_idx
        ]
        if strict and (unknown_user_ids or unknown_item_ids):
            error_messages: list[str] = []
            if unknown_user_ids:
                error_messages.append(f"unknown user_ids: {unknown_user_ids[:5]}")
            if unknown_item_ids:
                error_messages.append(f"unknown item_ids: {unknown_item_ids[:5]}")
            raise ValueError("Prediction received " + "; ".join(error_messages) + ".")

        available_user_ids = [
            int(user_id) for user_id in resolved_user_ids if int(user_id) in model.user_id_to_idx
        ]
        available_item_ids = [
            int(item_id) for item_id in resolved_item_ids if int(item_id) in model.item_id_to_idx
        ]
        return available_user_ids, available_item_ids, resolved_top_k

    def score_top_k_for_user(
        self,
        model: PredictableTwoTower,
        *,
        user_id: int,
        item_embeddings: torch.Tensor,
        item_ids: list[int],
        top_k: int,
        excluded_item_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if user_id not in model.user_id_to_idx:
            return []

        excluded_item_ids = excluded_item_ids or set()
        candidate_positions = [
            position for position, item_id in enumerate(item_ids)
            if item_id not in excluded_item_ids
        ]
        if not candidate_positions:
            return []

        user_embedding = self._normalize_user_embedding(model.get_user_embedding(user_id))
        candidate_embeddings = self._normalize_item_embeddings(item_embeddings[candidate_positions])
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
        model: PredictableTwoTower,
        *,
        user_id: int,
        item_embeddings: torch.Tensor,
        item_ids: list[int],
        top_k: int,
        excluded_item_ids: set[int] | None = None,
    ) -> set[int]:
        return {
            item_id
            for item_id, _score in self.score_top_k_for_user(
                model,
                user_id=user_id,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
                top_k=top_k,
                excluded_item_ids=excluded_item_ids,
            )
        }

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

    @staticmethod
    def _normalize_user_embedding(user_embedding: torch.Tensor) -> torch.Tensor:
        normalized_user_embedding = user_embedding.reshape(1, -1)
        normalized_user_embedding = F.normalize(normalized_user_embedding, dim=-1)
        return normalized_user_embedding.squeeze(0)

    @staticmethod
    def _normalize_item_embeddings(item_embeddings: torch.Tensor) -> torch.Tensor:
        normalized_item_embeddings = item_embeddings
        if normalized_item_embeddings.ndim == 1:
            normalized_item_embeddings = normalized_item_embeddings.unsqueeze(0)
        return F.normalize(normalized_item_embeddings, dim=-1)
