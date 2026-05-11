from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossInputs:
    positive_scores: torch.Tensor
    negative_scores: torch.Tensor


@dataclass(frozen=True)
class LossResult:
    loss: torch.Tensor
