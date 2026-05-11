from __future__ import annotations

import torch
import torch.nn.functional as F

from twotower._src.training.losses._types import LossInputs, LossResult


def compute_bpr_loss(inputs: LossInputs) -> LossResult:
    loss = -F.logsigmoid(inputs.positive_scores - inputs.negative_scores).mean()
    return LossResult(loss=loss)
