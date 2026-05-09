# FitResult.plot() Implementation Design

**Goal:** Add a `plot()` method to `FitResult` that visualises training loss curves after fitting.

**Architecture:** Single method on `FitResult` in `twotower/_src/training/fit.py`. Matplotlib is a lazy optional import — raised with a helpful error if not installed. No new modules or dependencies required.

**Tech Stack:** Python, matplotlib (optional, user-installed)

---

## API

```python
result = model.fit(inputs, epochs=20)

result.plot()              # opens a matplotlib window (plt.show())
result.plot("loss.png")    # saves to file, no plt.show()
result.plot(Path("out/loss.svg"))  # pathlib.Path accepted
```

Signature: `FitResult.plot(path: str | Path | None = None) -> None`

---

## Chart

- Single figure with one axes.
- x-axis: epoch number (1-based, integer ticks).
- y-axis: loss value.
- `train_loss` line always drawn (red/solid).
- `valid_loss` line drawn only if present in at least one history record (blue/dashed).
- Legend shown only when both lines are present.
- Axis labels: `"Epoch"` (x), `"Loss"` (y). Title: `"Training Loss"`.

---

## Behaviour

| Call | Result |
|------|--------|
| `result.plot()` | draws figure, calls `plt.show()` |
| `result.plot("loss.png")` | draws figure, calls `plt.savefig(path)`, no `plt.show()` |
| matplotlib not installed | raises `ImportError`: `"matplotlib is required for plotting. Install it with: pip install matplotlib"` |
| `result.history` is empty | raises `ValueError`: `"Cannot plot: training history is empty."` |

---

## Implementation

Method added to `FitResult` in `twotower/_src/training/fit.py`. No new files.

```python
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

---

## Tests

Added to `tests/test_fit.py`, matplotlib mocked via `unittest.mock.patch`.

- `test_plot_calls_show_when_no_path` — `plot()` calls `plt.show()`
- `test_plot_saves_file_when_path_given` — `plot("loss.png")` calls `fig.savefig("loss.png")`, `plt.show()` not called
- `test_plot_omits_valid_loss_when_absent` — history without `valid_loss` → only one `ax.plot` call
- `test_plot_draws_valid_loss_when_present` — history with `valid_loss` → two `ax.plot` calls + legend
- `test_plot_raises_import_error_when_matplotlib_missing` — `ImportError` with install hint
- `test_plot_raises_value_error_on_empty_history` — `ValueError` on empty `history`
