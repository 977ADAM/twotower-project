from __future__ import annotations

from typing import Callable

from twotower._src.training.losses._types import LossInputs, LossResult
from twotower._src.training.losses.bpr_loss import compute_bpr_loss

LOSS_REGISTRY: dict[str, Callable[[LossInputs], LossResult]] = {
    "BPR": compute_bpr_loss,
}
