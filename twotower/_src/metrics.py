from __future__ import annotations


def user_recall(actual: set[int], predicted: set[int]) -> float:
    if not actual:
        return 0.0
    return len(actual & predicted) / len(actual)


def mean_recall(per_user_recalls: list[float]) -> float:
    if not per_user_recalls:
        return 0.0
    return sum(per_user_recalls) / len(per_user_recalls)
