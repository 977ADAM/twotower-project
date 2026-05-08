from __future__ import annotations

from typing import Protocol

import torch

from twotower._src.config import _Config


class _HasEmbeddings(Protocol):
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...
    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor: ...
    def retrieval_logits(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor: ...


class _HasIDMappings(Protocol):
    query_id_to_idx: dict[int, int]
    candidate_id_to_idx: dict[int, int]
    idx_to_query_id: list[int]
    idx_to_candidate_id: list[int]


class _HasConfig(Protocol):
    config: _Config
