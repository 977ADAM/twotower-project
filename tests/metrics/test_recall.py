from __future__ import annotations

from twotower._src.metrics import mean_recall, user_recall, MetricInputs, MetricResult


def test_user_recall_perfect_prediction():
    result = user_recall(MetricInputs(actual=frozenset({1, 2, 3}), predicted=frozenset({1, 2, 3})))
    assert isinstance(result, MetricResult)
    assert result.value == 1.0


def test_user_recall_no_overlap():
    result = user_recall(MetricInputs(actual=frozenset({1, 2}), predicted=frozenset({3, 4})))
    assert result.value == 0.0


def test_user_recall_partial():
    result = user_recall(MetricInputs(actual=frozenset({1, 2, 3}), predicted=frozenset({1, 4, 5})))
    assert abs(result.value - 1 / 3) < 1e-9


def test_user_recall_empty_actual_returns_zero():
    result = user_recall(MetricInputs(actual=frozenset(), predicted=frozenset({1, 2})))
    assert result.value == 0.0


def test_mean_recall_empty_list():
    assert mean_recall([]).value == 0.0


def test_mean_recall_averages_correctly():
    results = [MetricResult(value=0.0), MetricResult(value=1.0)]
    assert mean_recall(results).value == 0.5


def test_metric_registry_contains_recall_key():
    from twotower._src.metrics import METRIC_REGISTRY
    assert "recall" in METRIC_REGISTRY
    assert callable(METRIC_REGISTRY["recall"])
