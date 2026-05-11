from __future__ import annotations

from typing import Callable

from twotower._src.metrics._types import MetricInputs, MetricResult
from twotower._src.metrics.recall import mean_recall, user_recall

METRIC_REGISTRY: dict[str, Callable[[MetricInputs], MetricResult]] = {
    "recall": user_recall,
}
