from __future__ import annotations

import importlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from twotower._src.config import _Config
from twotower._src.protocols import _HasConfig, _HasEmbeddings, _HasIDMappings
from twotower._src.training.losses import LOSS_REGISTRY, LossInputs, LossResult
from twotower._src.training.progress import EpochProgress, EpochSummary



@dataclass(slots=True, frozen=True)
class NegativeSampling:
    """Negative sampling strategy passed to fit().

    Controls how negative examples are drawn during training.
    `observed_ratio` sets the fraction of negatives sampled from
    observed non-positive interactions (vs random items).
    `in_batch_loss_weight` adds an InfoNCE contrastive loss over the
    batch similarity matrix on top of BPR; set to 0.0 to disable.
    """

    observed_ratio: float = 0.8
    in_batch_loss_weight: float = 0.0


@dataclass(slots=True, frozen=True)
class EarlyStopping:
    """Early stopping strategy passed to fit().

    Monitors `metric` after every epoch and halts training when it does not
    improve by more than `min_delta` for `patience` consecutive epochs.
    Metric mode (min/max) is inferred automatically: loss metrics use min,
    recall metrics use max.
    """

    patience: int = 5
    metric: str = "valid_loss"
    min_delta: float = 1e-4

    @property
    def mode(self) -> str:
        return "max" if self.metric.startswith("recall_at_") else "min"

    def is_better(self, current: float, best: float) -> bool:
        if self.mode == "max":
            return current > best + self.min_delta
        return current < best - self.min_delta


@dataclass(slots=True)
class FitInputs:
    """Prepared train/validation data for the training loop."""

    train_positive_df: pd.DataFrame
    train_interactions_df: pd.DataFrame
    num_users: int
    num_items: int
    valid_positive_df: pd.DataFrame | None = None
    valid_interactions_df: pd.DataFrame | None = None


@dataclass(slots=True)
class FitState:
    """Mutable training state owned by the trainer."""

    epoch: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass(slots=True)
class FitResult:
    """Artifacts returned by the trainer after fitting."""

    history: list[dict[str, float]]

    def plot(self, path: str | Path | None = None) -> None:
        if not self.history:
            raise ValueError("Cannot plot: training history is empty.")
        try:
            plt = importlib.import_module('matplotlib.pyplot')
            MaxNLocator = importlib.import_module('matplotlib.ticker').MaxNLocator
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. Install it with: pip install matplotlib"
            )

        epochs = [r["epoch"] for r in self.history]
        train_losses = [r["train_loss"] for r in self.history]
        valid_epochs = [r["epoch"] for r in self.history if r.get("valid_loss") is not None]
        valid_losses = [r["valid_loss"] for r in self.history if r.get("valid_loss") is not None]

        fig, ax = plt.subplots()
        ax.plot(epochs, train_losses, label="train_loss", color="tab:red")
        if valid_losses:
            ax.plot(valid_epochs, valid_losses, label="valid_loss", color="tab:blue", linestyle="--")
            ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        try:
            if path is None:
                plt.show()
            else:
                fig.savefig(path)
        finally:
            plt.close(fig)


class PairwiseInteractionsDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Yield (user, positive item, negative item) triples for BPR training."""

    def __init__(
        self,
        *,
        positive_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        query_id_to_idx: dict[int, int],
        candidate_id_to_idx: dict[int, int],
        num_items: int,
        observed_negative_sampling_ratio: float,
        seed: int,
    ):
        if not 0.0 <= observed_negative_sampling_ratio <= 1.0:
            raise ValueError("`observed_negative_sampling_ratio` must be in [0, 1].")

        self.user_tensor = torch.tensor(
            positive_df["query_id"].map(query_id_to_idx).to_numpy(),
            dtype=torch.long,
        )
        self.pos_item_tensor = torch.tensor(
            positive_df["candidate_id"].map(candidate_id_to_idx).to_numpy(),
            dtype=torch.long,
        )
        self.num_items = int(num_items)
        self.observed_negative_sampling_ratio = observed_negative_sampling_ratio
        self.random = random.Random(seed)
        self.observed_negative_items_by_user = self._build_item_pools_by_user(
            interactions_df=interactions_df,
            target_label=0.0,
            query_id_to_idx=query_id_to_idx,
            candidate_id_to_idx=candidate_id_to_idx,
            deduplicate=False,
        )
        positive_item_lists = self._build_item_pools_by_user(
            interactions_df=interactions_df,
            target_label=1.0,
            query_id_to_idx=query_id_to_idx,
            candidate_id_to_idx=candidate_id_to_idx,
            deduplicate=True,
        )
        self.positive_items_by_user = {
            user_idx: set(item_indices)
            for user_idx, item_indices in positive_item_lists.items()
        }

    def __len__(self) -> int:
        return int(self.user_tensor.size(0))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        user_idx = int(self.user_tensor[index].item())
        negative_item_idx = self._sample_negative_item(user_idx)
        return (
            self.user_tensor[index],
            self.pos_item_tensor[index],
            torch.tensor(negative_item_idx, dtype=torch.long),
        )

    def _sample_negative_item(self, user_idx: int) -> int:
        observed_negatives = self.observed_negative_items_by_user.get(user_idx, [])
        if observed_negatives and self.random.random() < self.observed_negative_sampling_ratio:
            return int(observed_negatives[self.random.randrange(len(observed_negatives))])

        positive_items = self.positive_items_by_user.get(user_idx, set())
        if len(positive_items) >= self.num_items:
            if observed_negatives:
                return int(observed_negatives[self.random.randrange(len(observed_negatives))])
            return 0

        while True:
            candidate_item_idx = self.random.randrange(self.num_items)
            if candidate_item_idx not in positive_items:
                return int(candidate_item_idx)

    @staticmethod
    def _build_item_pools_by_user(
        *,
        interactions_df: pd.DataFrame,
        target_label: float,
        query_id_to_idx: dict[int, int],
        candidate_id_to_idx: dict[int, int],
        deduplicate: bool,
    ) -> dict[int, list[int]]:
        filtered_df = interactions_df[interactions_df["label"] == target_label]
        pools: dict[int, list[int]] = {}
        for query_id, item_series in filtered_df.groupby("query_id")["candidate_id"]:
            if int(query_id) not in query_id_to_idx:
                continue

            mapped_items = [
                int(candidate_id_to_idx[int(candidate_id)])
                for candidate_id in item_series.astype(int).tolist()
                if int(candidate_id) in candidate_id_to_idx
            ]
            if deduplicate:
                mapped_items = list(dict.fromkeys(mapped_items))
            pools[int(query_id_to_idx[int(query_id)])] = mapped_items
        return pools


def build_pairwise_loader(
    *,
    positive_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    query_id_to_idx: dict[int, int],
    candidate_id_to_idx: dict[int, int],
    num_items: int,
    batch_size: int,
    shuffle: bool,
    observed_negative_sampling_ratio: float,
    seed: int,
    drop_last: bool = False,
) -> DataLoader[Any]:
    dataset = PairwiseInteractionsDataset(
        positive_df=positive_df,
        interactions_df=interactions_df,
        query_id_to_idx=query_id_to_idx,
        candidate_id_to_idx=candidate_id_to_idx,
        num_items=num_items,
        observed_negative_sampling_ratio=observed_negative_sampling_ratio,
        seed=seed,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


class _Trainable(_HasEmbeddings, _HasIDMappings, _HasConfig, Protocol):
    """Minimal model contract required by the training module."""

    def build_towers(self, num_users: int, num_items: int) -> None: ...

    def to(self, device: torch.device) -> nn.Module: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None: ...

    def train(self, mode: bool = True) -> object: ...

    def eval(self) -> object: ...

    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float: ...


class TwoTowerTrainer:
    """Train a two-tower model from prepared interaction data."""

    def __init__(self, config: _Config, device: torch.device):
        self.config = config
        self.device = device

    def fit(
        self,
        model: _Trainable,
        inputs: FitInputs,
        negative_sampling: NegativeSampling = NegativeSampling(),
        early_stopping: EarlyStopping | None = EarlyStopping(),
        loss_fn: str = "BPR",
    ) -> FitResult:
        """Run the full training loop and return training artifacts."""
        model.build_towers(inputs.num_users, inputs.num_items)
        model.to(self.device)

        has_validation = inputs.valid_positive_df is not None and inputs.valid_interactions_df is not None

        train_loader = self.build_train_loader(model, inputs, negative_sampling)
        valid_loader = self.build_valid_loader(model, inputs, negative_sampling) if has_validation else None
        optimizer = self.build_optimizer(model)
        if loss_fn not in LOSS_REGISTRY:
            raise ValueError(
                f"Unknown loss_fn {loss_fn!r}. "
                f"Available: {sorted(LOSS_REGISTRY)}"
            )
        loss_func = LOSS_REGISTRY[loss_fn]

        state = FitState()
        best_metric_value: float | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        need_recall = has_validation and (
            self.config.eval_during_training or (
                early_stopping is not None and early_stopping.metric.startswith("recall_at_")
            )
        )

        with EpochProgress() as progress:
            for epoch in range(1, self.config.epochs + 1):
                progress.start_epoch(epoch, self.config.epochs, num_batches=len(train_loader))
                train_metrics = self.train_epoch(
                    model=model,
                    train_loader=train_loader,
                    optimizer=optimizer,
                    loss_func=loss_func,
                    negative_sampling=negative_sampling,
                    progress=progress,
                )
                valid_metrics = self.validate(model=model, valid_loader=valid_loader, loss_func=loss_func)
                recall_metrics = self.compute_recall_metrics(model, inputs) if need_recall else {}
                epoch_metrics = self.merge_epoch_metrics(
                    epoch=epoch,
                    train_metrics=train_metrics,
                    valid_metrics=valid_metrics,
                    recall_metrics=recall_metrics,
                )
                state.epoch = epoch
                state.history.append(epoch_metrics)

                if early_stopping is not None:
                    current_value = epoch_metrics.get(early_stopping.metric)
                    if current_value is None:
                        raise ValueError(
                            f"Early stopping metric '{early_stopping.metric}' not found in epoch metrics. "
                            f"Available: {list(epoch_metrics.keys())}"
                        )

                    if best_metric_value is None or early_stopping.is_better(current_value, best_metric_value):
                        best_metric_value = current_value
                        best_state_dict = {
                            name: tensor.detach().cpu().clone()
                            for name, tensor in model.state_dict().items()
                        }
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                summary = EpochSummary(
                    metrics=epoch_metrics,
                    is_best=(epochs_without_improvement == 0 and best_metric_value is not None),
                    patience_used=epochs_without_improvement,
                    patience_total=early_stopping.patience if early_stopping is not None else None,
                )
                progress.finish_epoch(epoch, self.config.epochs, summary)

                if early_stopping is not None and epochs_without_improvement >= early_stopping.patience:
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        return FitResult(history=state.history)

    def build_optimizer(self, model: _Trainable) -> torch.optim.Optimizer:
        """Create the optimizer used by the training loop."""
        return torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def build_train_loader(
        self,
        model: _Trainable,
        inputs: FitInputs,
        negative_sampling: NegativeSampling,
    ) -> DataLoader[Any]:
        """Create the training dataloader."""
        return build_pairwise_loader(
            positive_df=inputs.train_positive_df,
            interactions_df=inputs.train_interactions_df,
            query_id_to_idx=model.query_id_to_idx,
            candidate_id_to_idx=model.candidate_id_to_idx,
            num_items=inputs.num_items,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            observed_negative_sampling_ratio=negative_sampling.observed_ratio,
            seed=self.config.seed,
        )

    def build_valid_loader(
        self,
        model: _Trainable,
        inputs: FitInputs,
        negative_sampling: NegativeSampling,
    ) -> DataLoader[Any]:
        """Create the validation dataloader."""
        assert inputs.valid_positive_df is not None and inputs.valid_interactions_df is not None
        return build_pairwise_loader(
            positive_df=inputs.valid_positive_df,
            interactions_df=inputs.valid_interactions_df,
            query_id_to_idx=model.query_id_to_idx,
            candidate_id_to_idx=model.candidate_id_to_idx,
            num_items=inputs.num_items,
            batch_size=self.config.batch_size,
            shuffle=False,
            observed_negative_sampling_ratio=negative_sampling.observed_ratio,
            seed=self.config.seed + 1,
        )

    def train_epoch(
        self,
        model: _Trainable,
        train_loader: DataLoader[Any],
        optimizer: torch.optim.Optimizer,
        loss_func: Callable[[LossInputs], LossResult],
        negative_sampling: NegativeSampling,
        progress: EpochProgress | None = None,
    ) -> dict[str, float]:
        """Run one training epoch and return train metrics."""
        model.train()
        loss_sum = 0.0
        total_examples = 0
        use_in_batch = negative_sampling.in_batch_loss_weight > 0.0

        for user_batch, pos_item_batch, neg_item_batch in train_loader:
            user_batch = user_batch.to(self.device)
            pos_item_batch = pos_item_batch.to(self.device)
            neg_item_batch = neg_item_batch.to(self.device)

            optimizer.zero_grad()

            positive_scores = model.score_pairs(user_batch, pos_item_batch)
            negative_scores = model.score_pairs(user_batch, neg_item_batch)
            result = loss_func(LossInputs(
                positive_scores=positive_scores,
                negative_scores=negative_scores,
            ))
            loss = result.loss

            if use_in_batch:
                logits = model.retrieval_logits(user_batch, pos_item_batch)
                labels = torch.arange(user_batch.size(0), device=self.device)
                in_batch: torch.Tensor = torch.nn.functional.cross_entropy(logits, labels)
                loss = loss + negative_sampling.in_batch_loss_weight * in_batch

            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            batch_size = user_batch.size(0)
            loss_sum += loss.item() * batch_size
            total_examples += batch_size

            if progress is not None:
                progress.advance()

        return {
            "train_loss": loss_sum / max(total_examples, 1),
        }

    def validate(
        self,
        model: _Trainable,
        valid_loader: DataLoader[Any] | None,
        loss_func: Callable[[LossInputs], LossResult],
    ) -> dict[str, float]:
        """Run validation and return validation metrics. Returns {} if no validation loader."""
        if valid_loader is None:
            return {}

        model.eval()
        loss_sum = 0.0
        total_examples = 0

        with torch.no_grad():
            for user_batch, pos_item_batch, neg_item_batch in valid_loader:
                user_batch = user_batch.to(self.device)
                pos_item_batch = pos_item_batch.to(self.device)
                neg_item_batch = neg_item_batch.to(self.device)

                positive_scores = model.score_pairs(user_batch, pos_item_batch)
                negative_scores = model.score_pairs(user_batch, neg_item_batch)
                result = loss_func(LossInputs(
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                ))
                loss = result.loss

                batch_size = user_batch.size(0)
                loss_sum += loss.item() * batch_size
                total_examples += batch_size

        return {
            "valid_loss": loss_sum / max(total_examples, 1),
        }

    def compute_recall_metrics(
        self,
        model: _Trainable,
        inputs: FitInputs,
    ) -> dict[str, float]:
        """Compute recall@k on the validation set for all configured top-k values."""
        eval_top_ks: list[int] = []
        for k in self.config.eval_top_ks:
            if int(k) not in eval_top_ks:
                eval_top_ks.append(int(k))

        metrics: dict[str, float] = {}
        for k in eval_top_ks:
            metrics[f"recall_at_{k}"] = model.recall_at_k(inputs.valid_interactions_df, k, exclude_seen=True)
        return metrics

    def merge_epoch_metrics(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        valid_metrics: dict[str, float],
        recall_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Merge train, validation, and recall metrics into one history record."""
        return {
            "epoch": float(epoch),
            **train_metrics,
            **valid_metrics,
            **recall_metrics,
        }

