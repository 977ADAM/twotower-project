from __future__ import annotations

import torch.nn as nn

from twotower._src.training.losses._types import LossInputs, LossResult


def compute_bpr_loss(inputs: LossInputs) -> LossResult:
    criterion = nn.LogSigmoid()
    loss = -criterion(inputs.positive_scores - inputs.negative_scores).mean()
    return LossResult(loss=loss)
