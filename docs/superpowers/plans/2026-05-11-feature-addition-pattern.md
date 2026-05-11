# Feature Addition Pattern Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Codify the feature addition pattern by migrating `compute_bpr_loss` and the recall metrics to the new module structure, introducing `LossInputs`/`LossResult`/`MetricInputs`/`MetricResult` DTOs, `LOSS_REGISTRY`, `METRIC_REGISTRY`, and wiring `loss_fn: str = "BPR"` through `TwoTower.fit()`.

**Architecture:** Two parallel tracks — losses (`training/losses/`) and metrics (`metrics/`) — each gets a `_types.py` for DTOs, one file per implementation, and a registry in `__init__.py`. The trainer resolves the loss by string at call time. The existing flat `metrics.py` is replaced by the new package; the import path stays identical so `evaluate.py` needs only call-site updates.

**Tech Stack:** Python, dataclasses, `typing.Callable`, `torch.nn`, pytest, `unittest.mock`

---

## File Map

| File | Action |
|------|--------|
| `twotower/_src/training/losses/_types.py` | Create — `LossInputs`, `LossResult` |
| `twotower/_src/training/losses/bpr_loss.py` | Create — `compute_bpr_loss` (moved + refactored from `fit.py`) |
| `twotower/_src/training/losses/__init__.py` | Create — `LOSS_REGISTRY` |
| `twotower/_src/training/fit.py` | Modify — remove `compute_bpr_loss`, `build_loss`; accept `loss_fn: str`; thread through `train_epoch`/`validate` |
| `twotower/_src/core.py` | Modify — add `loss_fn: str = "BPR"` to `TwoTower.fit()` |
| `twotower/_src/metrics/_types.py` | Create — `MetricInputs`, `MetricResult` |
| `twotower/_src/metrics/recall.py` | Create — `user_recall`, `mean_recall` (moved + refactored from `metrics.py`) |
| `twotower/_src/metrics/__init__.py` | Create — `METRIC_REGISTRY`, re-exports |
| `twotower/_src/metrics.py` | Delete — replaced by `metrics/` package |
| `twotower/_src/retrieval/evaluate.py` | Modify — update call sites to use `MetricInputs` / `.value` |
| `tests/training/__init__.py` | Create — empty, makes directory a package |
| `tests/training/losses/__init__.py` | Create — empty |
| `tests/training/losses/test_bpr_loss.py` | Create — unit tests for `compute_bpr_loss` |
| `tests/metrics/__init__.py` | Create — empty |
| `tests/metrics/test_recall.py` | Create — unit tests for `user_recall`, `mean_recall` |

---

## Task 1: Loss DTOs + bpr_loss.py + LOSS_REGISTRY

**Files:**
- Create: `twotower/_src/training/losses/_types.py`
- Create: `twotower/_src/training/losses/bpr_loss.py`
- Create: `twotower/_src/training/losses/__init__.py`
- Create: `tests/training/__init__.py`, `tests/training/losses/__init__.py`
- Create: `tests/training/losses/test_bpr_loss.py`

- [ ] **Step 1: Confirm baseline**

```bash
.venv/bin/pytest tests/ -q
```

Expected: 126 passed.

- [ ] **Step 2: Write the failing unit tests**

Create `tests/training/__init__.py` (empty) and `tests/training/losses/__init__.py` (empty).

Create `tests/training/losses/test_bpr_loss.py`:

```python
from __future__ import annotations

import torch

from twotower._src.training.losses._types import LossInputs, LossResult
from twotower._src.training.losses.bpr_loss import compute_bpr_loss


def test_bpr_loss_is_lower_when_positive_score_is_higher():
    good = LossInputs(
        positive_scores=torch.tensor([2.0]),
        negative_scores=torch.tensor([0.0]),
    )
    bad = LossInputs(
        positive_scores=torch.tensor([0.0]),
        negative_scores=torch.tensor([2.0]),
    )
    assert compute_bpr_loss(good).loss.item() < compute_bpr_loss(bad).loss.item()


def test_bpr_loss_result_is_scalar_tensor():
    inputs = LossInputs(
        positive_scores=torch.tensor([1.0, 2.0]),
        negative_scores=torch.tensor([0.0, 0.5]),
    )
    result = compute_bpr_loss(inputs)
    assert isinstance(result, LossResult)
    assert result.loss.shape == torch.Size([])


def test_bpr_loss_registry_contains_bpr_key():
    from twotower._src.training.losses import LOSS_REGISTRY
    assert "BPR" in LOSS_REGISTRY
    assert callable(LOSS_REGISTRY["BPR"])
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/training/losses/test_bpr_loss.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'twotower._src.training.losses'`

- [ ] **Step 4: Create `training/losses/_types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossInputs:
    positive_scores: torch.Tensor
    negative_scores: torch.Tensor


@dataclass(frozen=True)
class LossResult:
    loss: torch.Tensor
```

- [ ] **Step 5: Create `training/losses/bpr_loss.py`**

```python
from __future__ import annotations

import torch.nn as nn

from twotower._src.training.losses._types import LossInputs, LossResult


def compute_bpr_loss(inputs: LossInputs) -> LossResult:
    criterion = nn.LogSigmoid()
    loss = -criterion(inputs.positive_scores - inputs.negative_scores).mean()
    return LossResult(loss=loss)
```

- [ ] **Step 6: Create `training/losses/__init__.py`**

```python
from __future__ import annotations

from typing import Callable

from twotower._src.training.losses._types import LossInputs, LossResult
from twotower._src.training.losses.bpr_loss import compute_bpr_loss

LOSS_REGISTRY: dict[str, Callable[[LossInputs], LossResult]] = {
    "BPR": compute_bpr_loss,
}
```

- [ ] **Step 7: Run tests to confirm they pass**

```bash
.venv/bin/pytest tests/training/losses/test_bpr_loss.py -v
```

Expected: 3 passed.

- [ ] **Step 8: Run full suite to confirm no regressions**

```bash
.venv/bin/pytest tests/ -q
```

Expected: 129 passed (126 + 3 new).

- [ ] **Step 9: Commit**

```bash
git add twotower/_src/training/losses/ tests/training/
git commit -m "feat: add LossInputs/LossResult DTOs, bpr_loss.py, and LOSS_REGISTRY"
```

---

## Task 2: Wire LOSS_REGISTRY into TwoTowerTrainer and TwoTower.fit()

**Files:**
- Modify: `twotower/_src/training/fit.py`
- Modify: `twotower/_src/core.py`

- [ ] **Step 1: Write the failing integration test**

Add to `tests/test_core.py` (the `small_interactions` fixture and `TwoTower` import are already there):

```python
def test_fit_uses_registered_loss_fn(small_interactions):
    from unittest.mock import MagicMock, patch
    import torch
    from twotower._src.training.losses._types import LossInputs, LossResult

    train, valid, _ = small_interactions
    model = TwoTower()

    mock_loss = MagicMock(return_value=LossResult(loss=torch.tensor(0.5, requires_grad=True)))
    with patch.dict("twotower._src.training.losses.LOSS_REGISTRY", {"MOCK": mock_loss}):
        model.fit(
            train, validation_data=valid,
            epochs=1, batch_size=8,
            eval_during_training=False, device="cpu", seed=0,
            patience=None, loss_fn="MOCK",
        )

    mock_loss.assert_called()
    call_arg = mock_loss.call_args[0][0]
    assert isinstance(call_arg, LossInputs)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
.venv/bin/pytest tests/test_core.py::test_fit_uses_registered_loss_fn -v
```

Expected: FAIL — `TypeError: fit() got an unexpected keyword argument 'loss_fn'`

- [ ] **Step 3: Update imports in `fit.py`**

Add to the import block at the top of `twotower/_src/training/fit.py`:

```python
from typing import Callable  # add to existing typing import
from twotower._src.training.losses import LOSS_REGISTRY
from twotower._src.training.losses._types import LossInputs, LossResult
```

Remove:
```python
from twotower._src.training.fit import compute_bpr_loss  # (if self-referencing — not present)
```

Also remove the existing top-level `compute_bpr_loss` function definition (lines 19–26):

```python
# DELETE this block:
def compute_bpr_loss(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Compute a pairwise Bayesian Personalized Ranking loss."""
    result: torch.Tensor = -criterion(positive_scores - negative_scores).mean()
    return result
```

- [ ] **Step 4: Update `TwoTowerTrainer.fit()` signature**

In `twotower/_src/training/fit.py`, update the `fit()` method of `TwoTowerTrainer`:

```python
# BEFORE
def fit(
    self,
    model: _Trainable,
    inputs: FitInputs,
    negative_sampling: NegativeSampling = NegativeSampling(),
    early_stopping: EarlyStopping | None = EarlyStopping(),
) -> FitResult:
    ...
    criterion = self.build_loss()
    ...
    train_metrics = self.train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        negative_sampling=negative_sampling,
        progress=progress,
    )
    valid_metrics = self.validate(model=model, valid_loader=valid_loader, criterion=criterion)
```

```python
# AFTER
def fit(
    self,
    model: _Trainable,
    inputs: FitInputs,
    negative_sampling: NegativeSampling = NegativeSampling(),
    early_stopping: EarlyStopping | None = EarlyStopping(),
    loss_fn: str = "BPR",
) -> FitResult:
    ...
    loss_func = LOSS_REGISTRY[loss_fn]
    ...
    train_metrics = self.train_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        negative_sampling=negative_sampling,
        progress=progress,
    )
    valid_metrics = self.validate(model=model, valid_loader=valid_loader, loss_func=loss_func)
```

- [ ] **Step 5: Update `train_epoch()` signature and body**

```python
# BEFORE
def train_epoch(
    self,
    model: _Trainable,
    train_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    negative_sampling: NegativeSampling,
    progress: EpochProgress | None,
) -> dict[str, float]:
    ...
    loss = compute_bpr_loss(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        criterion=criterion,
    )
```

```python
# AFTER
def train_epoch(
    self,
    model: _Trainable,
    train_loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    loss_func: Callable[[LossInputs], LossResult],
    negative_sampling: NegativeSampling,
    progress: EpochProgress | None,
) -> dict[str, float]:
    ...
    result = loss_func(LossInputs(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
    ))
    loss = result.loss
```

- [ ] **Step 6: Update `validate()` signature and body**

```python
# BEFORE
def validate(
    self,
    model: _Trainable,
    valid_loader: DataLoader[Any] | None,
    criterion: nn.Module,
) -> dict[str, float]:
    ...
    loss = compute_bpr_loss(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
        criterion=criterion,
    )
```

```python
# AFTER
def validate(
    self,
    model: _Trainable,
    valid_loader: DataLoader[Any] | None,
    loss_func: Callable[[LossInputs], LossResult],
) -> dict[str, float]:
    ...
    result = loss_func(LossInputs(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
    ))
    loss = result.loss
```

- [ ] **Step 7: Remove `build_loss()` from `TwoTowerTrainer`**

Delete the method entirely (currently lines 376–378):

```python
# DELETE:
def build_loss(self) -> nn.Module:
    """Create the retrieval loss."""
    return nn.LogSigmoid()
```

- [ ] **Step 8: Add `loss_fn` parameter to `TwoTower.fit()` in `core.py`**

```python
# BEFORE (core.py, TwoTower.fit signature)
def fit(
    self,
    train_df: pd.DataFrame,
    *,
    ...
    min_delta: float = 1e-4,
) -> FitResult:
```

```python
# AFTER
def fit(
    self,
    train_df: pd.DataFrame,
    *,
    ...
    min_delta: float = 1e-4,
    loss_fn: str = "BPR",
) -> FitResult:
```

Also update the `trainer.fit()` call inside `TwoTower.fit()`:

```python
# BEFORE
fit_result = trainer.fit(
    self, fit_inputs,
    negative_sampling=negative_sampling,
    early_stopping=early_stopping,
)
```

```python
# AFTER
fit_result = trainer.fit(
    self, fit_inputs,
    negative_sampling=negative_sampling,
    early_stopping=early_stopping,
    loss_fn=loss_fn,
)
```

- [ ] **Step 9: Run full test suite**

```bash
.venv/bin/pytest tests/ -q
```

Expected: 130 passed (129 + 1 new integration test).

- [ ] **Step 10: Commit**

```bash
git add twotower/_src/training/fit.py twotower/_src/core.py tests/test_core.py
git commit -m "feat: wire LOSS_REGISTRY into TwoTowerTrainer; add loss_fn param to TwoTower.fit()"
```

---

## Task 3: Metric DTOs + metrics package + update evaluate.py

**Files:**
- Create: `twotower/_src/metrics/_types.py`
- Create: `twotower/_src/metrics/recall.py`
- Create: `twotower/_src/metrics/__init__.py`
- Delete: `twotower/_src/metrics.py`
- Modify: `twotower/_src/retrieval/evaluate.py`
- Create: `tests/metrics/__init__.py`, `tests/metrics/test_recall.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/metrics/__init__.py` (empty).

Create `tests/metrics/test_recall.py`:

```python
from __future__ import annotations

from twotower._src.metrics._types import MetricInputs, MetricResult
from twotower._src.metrics.recall import mean_recall, user_recall


def test_user_recall_perfect_prediction():
    result = user_recall(MetricInputs(actual={1, 2, 3}, predicted={1, 2, 3}))
    assert isinstance(result, MetricResult)
    assert result.value == 1.0


def test_user_recall_no_overlap():
    result = user_recall(MetricInputs(actual={1, 2}, predicted={3, 4}))
    assert result.value == 0.0


def test_user_recall_partial():
    result = user_recall(MetricInputs(actual={1, 2, 3}, predicted={1, 4, 5}))
    assert abs(result.value - 1 / 3) < 1e-9


def test_user_recall_empty_actual_returns_zero():
    result = user_recall(MetricInputs(actual=set(), predicted={1, 2}))
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
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/metrics/test_recall.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'twotower._src.metrics._types'`

- [ ] **Step 3: Create `metrics/_types.py`**

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricInputs:
    actual: set[int]
    predicted: set[int]


@dataclass(frozen=True)
class MetricResult:
    value: float
```

- [ ] **Step 4: Create `metrics/recall.py`**

```python
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
```

- [ ] **Step 5: Create `metrics/__init__.py`**

```python
from __future__ import annotations

from typing import Callable

from twotower._src.metrics._types import MetricInputs, MetricResult
from twotower._src.metrics.recall import mean_recall, user_recall

METRIC_REGISTRY: dict[str, Callable[[MetricInputs], MetricResult]] = {
    "recall": user_recall,
}
```

- [ ] **Step 6: Run metric unit tests to confirm they pass**

```bash
.venv/bin/pytest tests/metrics/test_recall.py -v
```

Expected: 7 passed.

- [ ] **Step 7: Delete `twotower/_src/metrics.py`**

```bash
rm twotower/_src/metrics.py
```

Note: the import `from twotower._src.metrics import mean_recall, user_recall` in `evaluate.py` will now resolve to the new `metrics/__init__.py`, which re-exports both. The path doesn't change.

- [ ] **Step 8: Update call sites in `evaluate.py`**

In `twotower/_src/retrieval/evaluate.py`, find and update all calls to `user_recall` and `mean_recall`.

Add import for `MetricInputs`:
```python
# BEFORE (line 11)
from twotower._src.metrics import mean_recall, user_recall
```

```python
# AFTER
from twotower._src.metrics import mean_recall, user_recall
from twotower._src.metrics._types import MetricInputs, MetricResult
```

In `recall_at_k` (two occurrences — one in the per-user loop, one at the return):

```python
# BEFORE (in the for-loop body)
recalls.append(user_recall(actual_items, predicted_items))
...
return mean_recall(recalls)
```

```python
# AFTER
recalls.append(user_recall(MetricInputs(actual=actual_items, predicted=predicted_items)))
...
return mean_recall(recalls).value
```

In `popularity_recall_at_k` (same pattern):

```python
# BEFORE
recalls.append(user_recall(actual_items, set(predicted_items)))
...
return mean_recall(recalls)
```

```python
# AFTER
recalls.append(user_recall(MetricInputs(actual=actual_items, predicted=set(predicted_items))))
...
return mean_recall(recalls).value
```

Also update the `recalls` type annotation if present from `list[float]` to `list[MetricResult]`.

- [ ] **Step 9: Run full test suite**

```bash
.venv/bin/pytest tests/ -q
```

Expected: 137 passed (130 + 7 new metric tests). All existing evaluate tests must pass.

- [ ] **Step 10: Commit**

```bash
git add twotower/_src/metrics/ twotower/_src/retrieval/evaluate.py tests/metrics/
git rm twotower/_src/metrics.py
git commit -m "feat: add MetricInputs/MetricResult DTOs, metrics package, METRIC_REGISTRY; update evaluate.py"
```
