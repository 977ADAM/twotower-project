from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricInputs:
    actual: frozenset[int]
    predicted: frozenset[int]


@dataclass(frozen=True)
class MetricResult:
    value: float
