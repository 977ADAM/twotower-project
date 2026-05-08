# Epoch Progress Bar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `EpochProgress` with a best-epoch marker, metric trend arrows, early stopping countdown, and a second overall-epoch progress bar.

**Architecture:** Add `EpochSummary` dataclass and a pure `_build_epoch_line()` helper to `progress.py`; update `EpochProgress` to manage two Rich tasks; update `fit.py` to construct `EpochSummary` and pass it to `finish_epoch`.

**Tech Stack:** Python 3.11, PyTorch, Rich, dataclasses, pytest

---

## File Structure

| File | Change |
|---|---|
| `twotower/_src/training/progress.py` | Add `EpochSummary`, `_build_epoch_line`, `_trend_markup`; update `EpochProgress` |
| `twotower/_src/training/fit.py` | Import `EpochSummary`; build it in `fit()`; remove lone `console.print` for early stopping |
| `tests/test_progress.py` | New file — tests for `EpochSummary` and `_build_epoch_line` |

---

### Task 1: `EpochSummary` dataclass

**Files:**
- Modify: `twotower/_src/training/progress.py`
- Create: `tests/test_progress.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_progress.py`:

```python
from __future__ import annotations

from twotower._src.training.progress import EpochSummary


def test_epoch_summary_defaults():
    s = EpochSummary(metrics={"train_loss": 0.5})
    assert s.is_best is False
    assert s.patience_used == 0
    assert s.patience_total is None


def test_epoch_summary_all_fields():
    s = EpochSummary(metrics={"train_loss": 0.5}, is_best=True, patience_used=2, patience_total=5)
    assert s.is_best is True
    assert s.patience_used == 2
    assert s.patience_total == 5
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/adam/projects/twotower-project
uv run pytest tests/test_progress.py -v
```

Expected: `ImportError` — `EpochSummary` not yet defined.

- [ ] **Step 3: Add `EpochSummary` to `progress.py`**

Add `dataclass` to the imports at the top of `progress.py`. The file currently imports from `rich.*` only — add this line:

```python
from dataclasses import dataclass
```

Then add the dataclass right after `console = Console()`:

```python
@dataclass(slots=True, frozen=True)
class EpochSummary:
    metrics: dict[str, float]
    is_best: bool = False
    patience_used: int = 0
    patience_total: int | None = None
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_progress.py -v
```

Expected: 2 PASSED.

- [ ] **Step 5: Commit**

```bash
git add twotower/_src/training/progress.py tests/test_progress.py
git commit -m "feat: add EpochSummary dataclass"
```

---

### Task 2: Pure rendering helper `_build_epoch_line`

**Files:**
- Modify: `twotower/_src/training/progress.py`
- Modify: `tests/test_progress.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_progress.py`:

```python
from twotower._src.training.progress import _build_epoch_line


def test_no_prev_metrics_shows_no_arrows():
    summary = EpochSummary(metrics={"train_loss": 0.5, "valid_loss": 0.4})
    line = _build_epoch_line(3, 5, summary, prev_metrics={})
    assert "↑" not in line
    assert "↓" not in line
    assert "→" not in line
    assert "0.5000" in line
    assert "0.4000" in line


def test_loss_decrease_shows_green_down_arrow():
    summary = EpochSummary(metrics={"train_loss": 0.4})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"train_loss": 0.5})
    assert "↓" in line
    assert "green" in line


def test_loss_increase_shows_red_up_arrow():
    summary = EpochSummary(metrics={"train_loss": 0.6})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"train_loss": 0.5})
    assert "↑" in line
    assert "red" in line


def test_recall_increase_shows_green_up_arrow():
    summary = EpochSummary(metrics={"recall_at_50": 0.6})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"recall_at_50": 0.5})
    assert "↑" in line
    assert "green" in line


def test_recall_decrease_shows_red_down_arrow():
    summary = EpochSummary(metrics={"recall_at_50": 0.4})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"recall_at_50": 0.5})
    assert "↓" in line
    assert "red" in line


def test_tiny_change_shows_dim_neutral_arrow():
    summary = EpochSummary(metrics={"train_loss": 0.50001})
    line = _build_epoch_line(2, 5, summary, prev_metrics={"train_loss": 0.5})
    assert "→" in line
    assert "dim" in line


def test_is_best_shows_star_marker():
    summary = EpochSummary(metrics={"train_loss": 0.4}, is_best=True)
    line = _build_epoch_line(2, 5, summary, prev_metrics={})
    assert "★" in line
    assert "best" in line


def test_patience_counter_shown_when_positive():
    summary = EpochSummary(metrics={"train_loss": 0.5}, patience_used=2, patience_total=5)
    line = _build_epoch_line(2, 5, summary, prev_metrics={})
    assert "no improvement" in line
    assert "2/5" in line


def test_patience_counter_hidden_when_zero():
    summary = EpochSummary(metrics={"train_loss": 0.5}, patience_used=0, patience_total=5)
    line = _build_epoch_line(2, 5, summary, prev_metrics={})
    assert "no improvement" not in line


def test_patience_counter_hidden_when_no_early_stopping():
    summary = EpochSummary(metrics={"train_loss": 0.5}, patience_used=3, patience_total=None)
    line = _build_epoch_line(2, 5, summary, prev_metrics={})
    assert "no improvement" not in line
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_progress.py -v
```

Expected: `ImportError` for `_build_epoch_line`.

- [ ] **Step 3: Implement `_build_epoch_line` and helpers in `progress.py`**

Add after `EpochSummary` and before `EpochProgress`:

```python
_TREND_EPSILON = 1e-4


def _is_metric_improvement(key: str, delta: float) -> bool:
    if key.startswith("recall_at_"):
        return delta > 0
    return delta < 0


def _trend_markup(key: str, delta: float) -> str:
    if abs(delta) < _TREND_EPSILON:
        return " [dim]→[/]"
    if _is_metric_improvement(key, delta):
        return f" [green]{'↓' if delta < 0 else '↑'}[/]"
    return f" [red]{'↑' if delta > 0 else '↓'}[/]"


def _build_epoch_line(
    epoch: int,
    total_epochs: int,
    summary: EpochSummary,
    prev_metrics: dict[str, float],
) -> str:
    parts: list[str] = [f"[dim]Epoch {epoch}/{total_epochs}[/]"]

    for key, value in summary.metrics.items():
        if key == "epoch":
            continue
        if key == "train_loss":
            part = f"[yellow]{key}[/]={value:.4f}"
        elif key == "valid_loss":
            part = f"[cyan]{key}[/]={value:.4f}"
        elif key.startswith("recall_at_"):
            part = f"[green]{key}[/]={value:.4f}"
        else:
            part = f"{key}={value:.4f}"

        if key in prev_metrics:
            part += _trend_markup(key, value - prev_metrics[key])

        parts.append(part)

    if summary.is_best:
        parts.append("[bold green]★ best[/]")

    if summary.patience_total is not None and summary.patience_used > 0:
        parts.append(f"[dim]no improvement {summary.patience_used}/{summary.patience_total}[/]")

    return "  ".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_progress.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add twotower/_src/training/progress.py tests/test_progress.py
git commit -m "feat: add _build_epoch_line with trend arrows, best marker, patience counter"
```

---

### Task 3: Update `EpochProgress` + integrate into `fit.py`

**Files:**
- Modify: `twotower/_src/training/progress.py`
- Modify: `twotower/_src/training/fit.py`

- [ ] **Step 1: Replace `EpochProgress` in `progress.py`**

Replace the entire `EpochProgress` class with:

```python
class EpochProgress:
    """Rich progress bar with two tracks: overall epoch progress and per-epoch batch progress.

    Usage::

        with EpochProgress() as bar:
            for epoch in range(1, total + 1):
                bar.start_epoch(epoch, total, num_batches=len(loader))
                for batch in loader:
                    # ... training step ...
                    bar.advance()
                bar.finish_epoch(epoch, total, summary)
    """

    def __init__(self) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]ETA"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )
        self._overall_task_id: TaskID | None = None
        self._batch_task_id: TaskID | None = None
        self._prev_metrics: dict[str, float] = {}

    def __enter__(self) -> "EpochProgress":
        self._progress.__enter__()
        self._overall_task_id = self._progress.add_task("Training", total=1)
        self._batch_task_id = self._progress.add_task("Epoch 0/0", total=1)
        return self

    def __exit__(self, *args: object) -> None:
        self._progress.__exit__(*args)

    def start_epoch(self, epoch: int, total_epochs: int, num_batches: int) -> None:
        """Reset the batch bar and update the overall bar description for a new epoch."""
        if self._overall_task_id is None or self._batch_task_id is None:
            return
        self._progress.update(
            self._overall_task_id,
            total=total_epochs,
            description=f"Training epoch {epoch}/{total_epochs}",
        )
        self._progress.reset(
            self._batch_task_id,
            total=num_batches,
            description=f"Epoch {epoch}/{total_epochs}",
        )

    def advance(self) -> None:
        """Advance the batch bar by one."""
        if self._batch_task_id is None:
            return
        self._progress.advance(self._batch_task_id)

    def finish_epoch(self, epoch: int, total_epochs: int, summary: EpochSummary) -> None:
        """Print a summary line and advance the overall epoch bar."""
        line = _build_epoch_line(epoch, total_epochs, summary, self._prev_metrics)
        self._progress.console.print(line)
        self._prev_metrics = dict(summary.metrics)
        if self._overall_task_id is not None:
            self._progress.advance(self._overall_task_id)
```

- [ ] **Step 2: Update `fit.py` — import `EpochSummary`**

In `twotower/_src/training/fit.py`, change the import line:

```python
from twotower._src.training.progress import EpochProgress
```

to:

```python
from twotower._src.training.progress import EpochProgress, EpochSummary
```

- [ ] **Step 3: Update `fit.py` — build `EpochSummary` and remove lone console.print**

In `TwoTowerTrainer.fit()`, find this block:

```python
                progress.finish_epoch(epoch, self.config.epochs, epoch_metrics)

                if early_stopping is not None and epochs_without_improvement >= early_stopping.patience:
                    console.print(
                        f"Early stopping at epoch {epoch} "
                        f"(no improvement in {early_stopping.metric} for {early_stopping.patience} epochs)"
                    )
                    break
```

Replace it with:

```python
                summary = EpochSummary(
                    metrics=epoch_metrics,
                    is_best=(epochs_without_improvement == 0 and best_metric_value is not None),
                    patience_used=epochs_without_improvement,
                    patience_total=early_stopping.patience if early_stopping is not None else None,
                )
                progress.finish_epoch(epoch, self.config.epochs, summary)

                if early_stopping is not None and epochs_without_improvement >= early_stopping.patience:
                    break
```

- [ ] **Step 4: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all existing tests PASSED, no regressions. The `test_progress.py` tests continue to pass.

- [ ] **Step 5: Commit**

```bash
git add twotower/_src/training/progress.py twotower/_src/training/fit.py
git commit -m "feat: two-bar epoch progress with trends, best marker, patience counter"
```
