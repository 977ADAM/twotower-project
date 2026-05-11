from __future__ import annotations

import torch.nn as nn

from twotower._src.training.losses._types import LossInputs, LossResult

_LOG_SIGMOID = nn.LogSigmoid()


def compute_bpr_loss(inputs: LossInputs) -> LossResult:
    loss = -_LOG_SIGMOID(inputs.positive_scores - inputs.negative_scores).mean()
    return LossResult(loss=loss)
