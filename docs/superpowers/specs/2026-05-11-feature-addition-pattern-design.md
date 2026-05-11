# Feature Addition Pattern Design

**Goal:** Establish a consistent internal standard for adding new loss functions, evaluation metrics, and other feature types to the TwoTower library.

**Scope:** Internal development pattern only — not an external plugin system.

**Tech Stack:** Python, dataclasses, typing.Callable, pytest

---

## Problem

There is currently no documented or codified answer to three questions that arise every time a new feature is added:

1. **Where does the code live?** — `compute_bpr_loss` sits in `fit.py`; metrics sit in a flat `metrics.py`; no convention signals where a new loss or metric should go.
2. **How does a new parameter reach the user?** — the boundary between `_Config` (hyperparameters) and `fit()` arguments (strategies) is implicit.
3. **How does new functionality connect to the service layer?** — no standard hook; each addition wires differently.

---

## Design

### 1. Module Structure

Each feature type gets its own package with one file per implementation:

```
twotower/_src/
  training/
    losses/
      __init__.py      ← LOSS_REGISTRY + public exports
      _types.py        ← LossInputs, LossResult
      bpr_loss.py      ← compute_bpr_loss (moved from fit.py)
      # new_loss.py    ← future losses: one file per loss
  metrics/
    __init__.py        ← METRIC_REGISTRY + public exports
    _types.py          ← MetricInputs, MetricResult
    recall.py          ← user_recall, mean_recall (moved from metrics.py)
    # ndcg.py          ← future metrics: one file per metric
```

**Rule:** one file per loss function, one file per metric. The file name is the signal — a new developer sees the folder and knows exactly where to add code.

---

### 2. DTO Contract

Each domain defines stable input/output types. Signatures never change when new fields are added — extend the DTO instead.

```python
# training/losses/_types.py
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

```python
# metrics/_types.py
from dataclasses import dataclass

@dataclass(frozen=True)
class MetricInputs:
    actual: set[int]
    predicted: set[int]

@dataclass(frozen=True)
class MetricResult:
    value: float
```

All loss functions have signature `(LossInputs) -> LossResult`.
All metric functions have signature `(MetricInputs) -> MetricResult`.

---

### 3. Registries

Each feature package owns a registry that maps string names to implementations.

```python
# training/losses/__init__.py
from twotower._src.training.losses._types import LossInputs, LossResult
from twotower._src.training.losses.bpr_loss import compute_bpr_loss

LOSS_REGISTRY: dict[str, Callable[[LossInputs], LossResult]] = {
    "BPR": compute_bpr_loss,
}
```

```python
# metrics/__init__.py
from twotower._src.metrics._types import MetricInputs, MetricResult
from twotower._src.metrics.recall import user_recall

METRIC_REGISTRY: dict[str, Callable[[MetricInputs], MetricResult]] = {
    "recall": user_recall,
}
```

**User-facing API — strings only:**
```python
model.fit(train_df, loss_fn="BPR")          # default
model.fit(train_df, loss_fn="BCE")          # new loss, registered
```

**Rule:** `_Config` holds hyperparameters (numbers, flags). Strategy choices (which loss, which metric) are string arguments resolved through the registry at call time — they do not go into `_Config`.

---

### 4. Service Wiring

`TwoTowerTrainer.fit()` resolves the loss from the registry and passes it to the training loop:

```python
# training/fit.py
from twotower._src.training.losses import LOSS_REGISTRY

class TwoTowerTrainer:
    def fit(self, model, ..., loss_fn: str = "BPR"):
        fn = LOSS_REGISTRY[loss_fn]
        ...
        result: LossResult = fn(LossInputs(
            positive_scores=positive_scores,
            negative_scores=negative_scores,
        ))
        loss = result.loss
```

If an unregistered name is passed, `LOSS_REGISTRY[loss_fn]` raises `KeyError` — the error message naturally lists available keys.

---

### 5. Testing Convention

**Both levels are required.** A feature without both levels of tests is not considered done.

#### Level 1 — Unit (pure function)

Test the implementation in isolation with simple tensors or Python builtins. No model, no fixtures.

```python
# tests/training/losses/test_bpr_loss.py
from twotower._src.training.losses.bpr_loss import compute_bpr_loss
from twotower._src.training.losses._types import LossInputs

def test_bpr_loss_is_lower_when_positive_score_is_higher():
    inputs = LossInputs(
        positive_scores=torch.tensor([2.0, 1.0]),
        negative_scores=torch.tensor([0.0, 0.5]),
    )
    reversed_inputs = LossInputs(
        positive_scores=inputs.negative_scores,
        negative_scores=inputs.positive_scores,
    )
    assert compute_bpr_loss(inputs).loss.item() < compute_bpr_loss(reversed_inputs).loss.item()
```

```python
# tests/metrics/test_recall.py
from twotower._src.metrics.recall import user_recall
from twotower._src.metrics._types import MetricInputs

def test_user_recall_perfect_prediction():
    result = user_recall(MetricInputs(actual={1, 2, 3}, predicted={1, 2, 3}))
    assert result.value == 1.0

def test_user_recall_no_overlap():
    result = user_recall(MetricInputs(actual={1, 2}, predicted={3, 4}))
    assert result.value == 0.0
```

#### Level 2 — Integration (via stub)

Test that the service layer resolves the registry name and calls the function.

```python
# tests/training/test_fit.py
from unittest.mock import MagicMock, patch
from twotower._src.training.losses._types import LossInputs, LossResult

def test_trainer_uses_registered_loss_fn(stub_trainable_model):
    mock_loss = MagicMock(return_value=LossResult(loss=torch.tensor(0.5)))
    with patch.dict("twotower._src.training.losses.LOSS_REGISTRY", {"MOCK": mock_loss}):
        trainer.fit(stub_trainable_model, ..., loss_fn="MOCK")
    mock_loss.assert_called()
    call_args = mock_loss.call_args[0][0]
    assert isinstance(call_args, LossInputs)
```

---

### 6. Checklist: Adding a New Loss Function

When adding, for example, `BCE` loss:

- [ ] Create `training/losses/bce_loss.py` with `compute_bce_loss(inputs: LossInputs) -> LossResult`
- [ ] Register in `training/losses/__init__.py`: `"BCE": compute_bce_loss`
- [ ] Write unit test in `tests/training/losses/test_bce_loss.py`
- [ ] Write integration test in `tests/training/test_fit.py` (or extend existing)
- [ ] Done — no changes to `TwoTower` public API

The same checklist applies for metrics, replacing `losses` with `metrics` throughout.

---

## Files Changed

| File | Change |
|------|--------|
| `twotower/_src/training/losses/__init__.py` | New — `LOSS_REGISTRY`, exports |
| `twotower/_src/training/losses/_types.py` | New — `LossInputs`, `LossResult` |
| `twotower/_src/training/losses/bpr_loss.py` | New — `compute_bpr_loss` moved from `fit.py` |
| `twotower/_src/training/fit.py` | Remove `compute_bpr_loss`; wire registry |
| `twotower/_src/metrics/__init__.py` | New — `METRIC_REGISTRY`, exports |
| `twotower/_src/metrics/_types.py` | New — `MetricInputs`, `MetricResult` |
| `twotower/_src/metrics/recall.py` | New — `user_recall`, `mean_recall` moved from `metrics.py` |
| `twotower/_src/metrics.py` | Delete (replaced by package) |
| `tests/training/losses/test_bpr_loss.py` | New — unit tests |
| `tests/metrics/test_recall.py` | New — unit tests |

---

## Non-Goals

- External plugin system (users registering their own loss functions from outside the library)
- Callable support in `loss_fn` / `metric_fn` — strings only
- Tower architecture extension pattern (separate future spec)
