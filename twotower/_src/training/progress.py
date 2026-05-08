from __future__ import annotations

from dataclasses import dataclass

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


class EpochProgress:
    """Rich progress bar tracking batch-level progress within each epoch.

    Usage::

        with EpochProgress() as bar:
            for epoch in range(1, total + 1):
                bar.start_epoch(epoch, total, num_batches=len(loader))
                for batch in loader:
                    # ... training step ...
                    bar.advance()
                bar.finish_epoch(epoch, total, metrics)
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
        self._task_id: TaskID | None = None

    def __enter__(self) -> "EpochProgress":
        self._progress.__enter__()
        self._task_id = self._progress.add_task("Epoch 0/0", total=1)
        return self

    def __exit__(self, *args: object) -> None:
        self._progress.__exit__(*args)

    def start_epoch(self, epoch: int, total_epochs: int, num_batches: int) -> None:
        """Reset the bar for a new epoch."""
        if self._task_id is None:
            return
        self._progress.reset(
            self._task_id,
            total=num_batches,
            description=f"Epoch {epoch}/{total_epochs}",
        )

    def advance(self) -> None:
        """Advance the bar by one batch."""
        if self._task_id is None:
            return
        self._progress.advance(self._task_id)

    def finish_epoch(self, epoch: int, total_epochs: int, metrics: dict[str, float]) -> None:
        """Print a summary line with metrics after the epoch completes."""
        parts: list[str] = []
        if "train_loss" in metrics:
            parts.append(f"[yellow]train_loss[/]={metrics['train_loss']:.4f}")
        if "valid_loss" in metrics:
            parts.append(f"[cyan]valid_loss[/]={metrics['valid_loss']:.4f}")
        for key, value in metrics.items():
            if key.startswith("recall_at_"):
                parts.append(f"[green]{key}[/]={value:.4f}")

        self._progress.console.print(
            f"[dim]Epoch {epoch}/{total_epochs}[/]  " + "  ".join(parts)
        )
