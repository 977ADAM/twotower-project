from __future__ import annotations

from twotower._src.metrics._types import MetricInputs, MetricResult


def user_recall(inputs: MetricInputs) -> MetricResult:
    if not inputs.actual:
        return MetricResult(value=0.0)
    return MetricResult(value=len(inputs.actual & inputs.predicted) / len(inputs.actual))


def mean_recall(per_user_results: list[MetricResult]) -> MetricResult:
    if not per_user_results:
        return MetricResult(value=0.0)
    return MetricResult(value=sum(r.value for r in per_user_results) / len(per_user_results))
