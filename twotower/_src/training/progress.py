from __future__ import annotations

from dataclasses import dataclass
from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


@dataclass(slots=True, frozen=True)
class EpochSummary:
    metrics: dict[str, float]
    is_best: bool = False
    patience_used: int = 0
    patience_total: int | None = None


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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._progress.__exit__(exc_type, exc_val, exc_tb)

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
