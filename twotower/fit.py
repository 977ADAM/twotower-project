from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd
import torch
import torch.nn as nn
from rich.console import Console
from torch.utils.data import DataLoader, TensorDataset

from twotower.config import TwoTowerConfig

console = Console()


def compute_in_batch_retrieval_loss(
    logits: torch.Tensor,
    criterion: nn.Module,
    symmetric: bool,
) -> torch.Tensor:
    """Compute a contrastive in-batch softmax loss for retrieval."""
    targets = torch.arange(logits.size(0), device=logits.device)
    loss = criterion(logits, targets)
    if symmetric:
        loss = 0.5 * (loss + criterion(logits.T, targets))
    return loss


@dataclass(slots=True)
class FitInputs:
    """Prepared train/validation data for the training loop."""

    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    num_users: int
    num_items: int


@dataclass(slots=True)
class FitState:
    """Mutable training state owned by the trainer."""

    epoch: int = 0
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass(slots=True)
class FitResult:
    """Artifacts returned by the trainer after fitting."""

    history: list[dict[str, float]]


class TrainableTwoTower(Protocol):
    """Minimal model contract required by the training module."""

    config: TwoTowerConfig
    user_id_to_idx: dict[int, int]
    item_id_to_idx: dict[int, int]

    def build_towers(self, num_users: int, num_items: int) -> None:
        ...

    def to(self, device: torch.device) -> nn.Module:
        ...

    def parameters(self):
        ...

    def state_dict(self) -> dict[str, torch.Tensor]:
        ...

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        ...

    def train(self, mode: bool = True):
        ...

    def eval(self):
        ...

    def score_pairs(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
    ) -> torch.Tensor:
        ...

    def retrieval_logits(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
    ) -> torch.Tensor:
        ...


class TwoTowerTrainer:
    """Train a two-tower model from prepared interaction data."""

    def __init__(self, config: TwoTowerConfig, device: torch.device):
        self.config = config
        self.device = device

    def fit(self, model: TrainableTwoTower, inputs: FitInputs) -> FitResult:
        """Run the full training loop and return training artifacts."""
        model.build_towers(inputs.num_users, inputs.num_items)
        model.to(self.device)

        train_loader = self.build_train_loader(model, inputs)
        valid_loader = self.build_valid_loader(model, inputs)
        optimizer = self.build_optimizer(model)
        criterion = self.build_loss()

        state = FitState()
        best_valid_loss: float | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
            )
            valid_metrics = self.validate(
                model=model,
                valid_loader=valid_loader,
                criterion=criterion,
            )
            epoch_metrics = self.merge_epoch_metrics(
                epoch=epoch,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )
            state.epoch = epoch
            state.history.append(epoch_metrics)

            current_valid_loss = epoch_metrics["valid_loss"]
            if best_valid_loss is None or current_valid_loss < best_valid_loss:
                best_valid_loss = current_valid_loss
                best_state_dict = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }

            console.print(
                f"Epoch {epoch}/{self.config.epochs} "
                f"train_loss={epoch_metrics['train_loss']:.4f} "
                f"valid_loss={epoch_metrics['valid_loss']:.4f}"
            )

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        return FitResult(history=state.history)

    def build_optimizer(self, model: TrainableTwoTower) -> torch.optim.Optimizer:
        """Create the optimizer used by the training loop."""
        return torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

    def build_loss(self) -> nn.Module:
        """Create the retrieval loss."""
        return nn.CrossEntropyLoss()

    def build_train_loader(
        self,
        model: TrainableTwoTower,
        inputs: FitInputs,
    ) -> DataLoader:
        """Create the training dataloader."""
        return self._make_loader(
            dataframe=inputs.train_df,
            user_id_to_idx=model.user_id_to_idx,
            item_id_to_idx=model.item_id_to_idx,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

    def build_valid_loader(
        self,
        model: TrainableTwoTower,
        inputs: FitInputs,
    ) -> DataLoader:
        """Create the validation dataloader."""
        return self._make_loader(
            dataframe=inputs.valid_df,
            user_id_to_idx=model.user_id_to_idx,
            item_id_to_idx=model.item_id_to_idx,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def train_epoch(
        self,
        model: TrainableTwoTower,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> dict[str, float]:
        """Run one training epoch and return train metrics."""
        model.train()
        loss_sum = 0.0
        total_examples = 0

        for user_batch, item_batch in train_loader:
            user_batch = user_batch.to(self.device)
            item_batch = item_batch.to(self.device)

            optimizer.zero_grad()
            logits = model.retrieval_logits(user_batch, item_batch)
            loss = compute_in_batch_retrieval_loss(
                logits=logits,
                criterion=criterion,
                symmetric=self.config.symmetric_retrieval_loss,
            )
            loss.backward()
            optimizer.step()

            batch_size = user_batch.size(0)
            loss_sum += loss.item() * batch_size
            total_examples += batch_size

        return {
            "train_loss": loss_sum / max(total_examples, 1),
        }

    def validate(
        self,
        model: TrainableTwoTower,
        valid_loader: DataLoader,
        criterion: nn.Module,
    ) -> dict[str, float]:
        """Run validation and return validation metrics."""
        model.eval()
        loss_sum = 0.0
        total_examples = 0

        with torch.no_grad():
            for user_batch, item_batch in valid_loader:
                user_batch = user_batch.to(self.device)
                item_batch = item_batch.to(self.device)

                logits = model.retrieval_logits(user_batch, item_batch)
                loss = compute_in_batch_retrieval_loss(
                    logits=logits,
                    criterion=criterion,
                    symmetric=self.config.symmetric_retrieval_loss,
                )

                batch_size = user_batch.size(0)
                loss_sum += loss.item() * batch_size
                total_examples += batch_size

        return {
            "valid_loss": loss_sum / max(total_examples, 1),
        }

    def merge_epoch_metrics(
        self,
        *,
        epoch: int,
        train_metrics: dict[str, float],
        valid_metrics: dict[str, float],
    ) -> dict[str, float]:
        """Merge train and validation metrics into one history record."""
        return {
            "epoch": float(epoch),
            **train_metrics,
            **valid_metrics,
        }

    def _make_loader(
        self,
        *,
        dataframe: pd.DataFrame,
        user_id_to_idx: dict[int, int],
        item_id_to_idx: dict[int, int],
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        user_tensor = torch.tensor(
            dataframe["user_id"].map(user_id_to_idx).to_numpy(),
            dtype=torch.long,
        )
        item_tensor = torch.tensor(
            dataframe["banner_id"].map(item_id_to_idx).to_numpy(),
            dtype=torch.long,
        )
        dataset = TensorDataset(user_tensor, item_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
