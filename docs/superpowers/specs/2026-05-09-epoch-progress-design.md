# Epoch Progress Bar — Design Spec

**Date:** 2026-05-09  
**Files affected:** `twotower/_src/training/progress.py`, `twotower/_src/training/fit.py`

---

## Goal

Extend `EpochProgress` with four improvements that make training more informative at a glance:

1. **★ best epoch marker** — highlight when a new best checkpoint is saved
2. **Metric trends** — ↑/↓/→ arrows showing change vs previous epoch
3. **Early stopping countdown** — `no improvement 2/5` counter in each epoch line
4. **Overall epoch progress bar** — second bar showing epoch X/N above the batch bar

---

## New Type: `EpochSummary`

Added to `progress.py`:

```python
@dataclass(slots=True, frozen=True)
class EpochSummary:
    metrics: dict[str, float]
    is_best: bool = False
    patience_used: int = 0
    patience_total: int | None = None  # None when early stopping is disabled
```

`EpochProgress` tracks `_prev_metrics: dict[str, float]` internally. Trend arrows are computed inside `finish_epoch` — callers do not supply previous metrics.

---

## `EpochProgress` Changes

### Two progress tasks

`Progress` manages two tasks:

| Task | Description | Advances |
|---|---|---|
| `_overall_task_id` | `Training epoch X/N` | Once per `finish_epoch` |
| `_batch_task_id` | `Epoch X/N` (batches) | Once per `advance()` |

Both tasks are created in `__enter__` with `total=1` as a placeholder. `start_epoch` calls `progress.update(_overall_task_id, total=total_epochs)` on the first call to set the correct total.

### Updated method signatures

```python
def start_epoch(self, epoch: int, total_epochs: int, num_batches: int) -> None: ...
def advance(self) -> None: ...
def finish_epoch(self, epoch: int, total_epochs: int, summary: EpochSummary) -> None: ...
```

`start_epoch` and `advance` signatures are unchanged. `finish_epoch` replaces the `metrics: dict[str, float]` parameter with `summary: EpochSummary`.

### Epoch summary line rendering

`finish_epoch` builds the printed line in this order:

1. **Epoch label** — `Epoch 4/5` in dim
2. **Metrics with trends** — for each metric, compare to `_prev_metrics`:
   - No previous value: show plain `metric=0.1234`
   - For loss metrics (`train_loss`, `valid_loss`): ↓ is green (good), ↑ is red (bad)
   - For recall metrics (`recall_at_*`): ↑ is green (good), ↓ is red (bad)
   - Change smaller than `1e-4` in absolute value: `→` in dim
3. **Best marker** — `★ best` in bold green if `summary.is_best`
4. **Patience counter** — `no improvement {patience_used}/{patience_total}` in dim, only shown when `patience_total is not None` and `patience_used > 0`

After printing, `_prev_metrics` is updated to `summary.metrics` and `_overall_task_id` advances by 1.

**Example output:**
```
Epoch 3/5  train_loss=0.4231  valid_loss=0.3892  recall_at_50=0.1234
Epoch 4/5  train_loss=0.3981 ↓  valid_loss=0.3750 ↓  recall_at_50=0.1456 ↑  ★ best
Epoch 5/5  train_loss=0.4100 ↑  valid_loss=0.3900 ↑  recall_at_50=0.1389 ↓  no improvement 1/5
```

---

## `fit.py` Changes

After early stopping state is updated, assemble `EpochSummary` and pass it to `finish_epoch`:

```python
summary = EpochSummary(
    metrics=epoch_metrics,
    is_best=(epochs_without_improvement == 0 and best_metric_value is not None),
    patience_used=epochs_without_improvement,
    patience_total=early_stopping.patience if early_stopping is not None else None,
)
progress.finish_epoch(epoch, self.config.epochs, summary)
```

The standalone `console.print(...)` early stopping message is removed — `no improvement N/N` in the final epoch line makes it redundant.

No other changes to `fit.py`.

---

## What is not changing

- `EarlyStopping`, `NegativeSampling`, `TwoTowerTrainer` logic — untouched
- The `console` module-level instance in `fit.py` — kept for the early stopping fallback (now removed, but console stays for future use)
- Public API (`TwoTower`, `split_interactions`) — unaffected
