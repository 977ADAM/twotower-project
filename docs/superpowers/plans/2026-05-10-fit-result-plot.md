# FitResult.plot() Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `plot()` method to `FitResult` that draws training loss curves and either shows them or saves to a file.

**Architecture:** Single method added to the existing `FitResult` dataclass in `twotower/_src/training/fit.py`. Matplotlib is imported lazily inside the method so it remains an optional dependency — users who never call `plot()` don't need it installed.

**Tech Stack:** Python, matplotlib (optional, user-installed), pytest + unittest.mock

---

## File Structure

- **Modify:** `twotower/_src/training/fit.py` — add `from pathlib import Path` import; add `plot()` method to `FitResult` (lines 87–92)
- **Modify:** `tests/test_fit.py` — append 6 new plot tests at end of file

---

### Task 1: Add `FitResult.plot()` and tests

**Files:**
- Modify: `twotower/_src/training/fit.py:1-5` (add `Path` import), `twotower/_src/training/fit.py:87-92` (add method)
- Modify: `tests/test_fit.py` (append 6 tests)

- [ ] **Step 1: Append 6 failing tests to `tests/test_fit.py`**

Add these imports at the top of the existing import block in `tests/test_fit.py` (after the existing imports):

```python
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from twotower._src.training.fit import FitResult
```

Then append the following at the **end** of `tests/test_fit.py`:

```python
# ---------------------------------------------------------------------------
# FitResult.plot() tests
# ---------------------------------------------------------------------------

def _make_mocks():
    mock_plt = MagicMock()
    mock_fig = MagicMock()
    mock_ax = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)
    return mock_plt, mock_fig, mock_ax


_HISTORY_NO_VALID = [
    {"epoch": 1.0, "train_loss": 0.5},
    {"epoch": 2.0, "train_loss": 0.4},
]

_HISTORY_WITH_VALID = [
    {"epoch": 1.0, "train_loss": 0.5, "valid_loss": 0.6},
    {"epoch": 2.0, "train_loss": 0.4, "valid_loss": 0.5},
]


def test_plot_calls_show_when_no_path():
    mock_plt, mock_fig, mock_ax = _make_mocks()
    with patch.dict(sys.modules, {"matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt}):
        FitResult(history=_HISTORY_NO_VALID).plot()
    mock_plt.show.assert_called_once()
    mock_fig.savefig.assert_not_called()


def test_plot_saves_file_when_path_given():
    mock_plt, mock_fig, mock_ax = _make_mocks()
    with patch.dict(sys.modules, {"matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt}):
        FitResult(history=_HISTORY_NO_VALID).plot("loss.png")
    mock_fig.savefig.assert_called_once_with("loss.png")
    mock_plt.show.assert_not_called()


def test_plot_omits_valid_loss_when_absent():
    mock_plt, mock_fig, mock_ax = _make_mocks()
    with patch.dict(sys.modules, {"matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt}):
        FitResult(history=_HISTORY_NO_VALID).plot()
    assert mock_ax.plot.call_count == 1
    mock_ax.legend.assert_not_called()


def test_plot_draws_valid_loss_when_present():
    mock_plt, mock_fig, mock_ax = _make_mocks()
    with patch.dict(sys.modules, {"matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt}):
        FitResult(history=_HISTORY_WITH_VALID).plot()
    assert mock_ax.plot.call_count == 2
    mock_ax.legend.assert_called_once()


def test_plot_raises_import_error_when_matplotlib_missing():
    with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
        with pytest.raises(ImportError, match="pip install matplotlib"):
            FitResult(history=_HISTORY_NO_VALID).plot()


def test_plot_raises_value_error_on_empty_history():
    mock_plt, mock_fig, mock_ax = _make_mocks()
    with patch.dict(sys.modules, {"matplotlib": MagicMock(), "matplotlib.pyplot": mock_plt}):
        with pytest.raises(ValueError, match="training history is empty"):
            FitResult(history=[]).plot()
```

- [ ] **Step 2: Run the new tests to confirm they all fail**

```bash
.venv/bin/pytest tests/test_fit.py::test_plot_calls_show_when_no_path \
  tests/test_fit.py::test_plot_saves_file_when_path_given \
  tests/test_fit.py::test_plot_omits_valid_loss_when_absent \
  tests/test_fit.py::test_plot_draws_valid_loss_when_present \
  tests/test_fit.py::test_plot_raises_import_error_when_matplotlib_missing \
  tests/test_fit.py::test_plot_raises_value_error_on_empty_history \
  -v
```

Expected: all 6 FAIL with `AttributeError: 'FitResult' object has no attribute 'plot'`

- [ ] **Step 3: Add `Path` import to `twotower/_src/training/fit.py`**

Change line 3 from:

```python
import random
```

to:

```python
import random
from pathlib import Path
```

- [ ] **Step 4: Add `plot()` method to `FitResult` in `twotower/_src/training/fit.py`**

Replace:

```python
@dataclass(slots=True)
class FitResult:
    """Artifacts returned by the trainer after fitting."""

    history: list[dict[str, float]]
```

with:

```python
@dataclass(slots=True)
class FitResult:
    """Artifacts returned by the trainer after fitting."""

    history: list[dict[str, float]]

    def plot(self, path: str | Path | None = None) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            )
        if not self.history:
            raise ValueError("Cannot plot: training history is empty.")

        epochs = [int(r["epoch"]) for r in self.history]
        train_losses = [r["train_loss"] for r in self.history]
        valid_epochs = [int(r["epoch"]) for r in self.history if r.get("valid_loss") is not None]
        valid_losses = [r["valid_loss"] for r in self.history if r.get("valid_loss") is not None]

        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, label="train_loss", color="tab:red")
        if valid_losses:
            ax.plot(valid_epochs, valid_losses, label="valid_loss", color="tab:blue", linestyle="--")
            ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.xaxis.get_major_locator().set_params(integer=True)

        if path is None:
            plt.show()
        else:
            fig.savefig(path)
        plt.close(fig)
```

- [ ] **Step 5: Run all 6 new tests to confirm they pass**

```bash
.venv/bin/pytest tests/test_fit.py::test_plot_calls_show_when_no_path \
  tests/test_fit.py::test_plot_saves_file_when_path_given \
  tests/test_fit.py::test_plot_omits_valid_loss_when_absent \
  tests/test_fit.py::test_plot_draws_valid_loss_when_present \
  tests/test_fit.py::test_plot_raises_import_error_when_matplotlib_missing \
  tests/test_fit.py::test_plot_raises_value_error_on_empty_history \
  -v
```

Expected: all 6 PASS

- [ ] **Step 6: Run the full test suite to check for regressions**

```bash
.venv/bin/pytest --tb=short -q
```

Expected: 125 passed (119 existing + 6 new), 0 failures

- [ ] **Step 7: Commit**

```bash
git add twotower/_src/training/fit.py tests/test_fit.py
git commit -m "feat: add FitResult.plot() for training loss visualisation"
```
