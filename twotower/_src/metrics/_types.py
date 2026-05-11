from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricInputs:
    actual: set[int]
    predicted: set[int]


@dataclass(frozen=True)
class MetricResult:
    value: float
