from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader, TensorDataset

from twotower.config import TwoTowerConfig
from twotower.user_tower import UserTower
from twotower.item_tower import ItemTower

console = Console()

class TwoTowerBase(nn.Module):
    def __init__(
        self,
        config: TwoTowerConfig,
        num_users: int | None = None,
        num_items: int | None = None,
    ):
        super().__init__()
        self.config = config
        self.user_tower: UserTower | None = None
        self.item_tower: ItemTower | None = None
        if num_users is not None and num_items is not None:
            self.build_towers(num_users, num_items)

    def build_towers(self, num_users: int, num_items: int) -> None:
        self.user_tower = UserTower(num_users, self.config)
        self.item_tower = ItemTower(num_items, self.config)

    def forward(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.user_tower is None or self.item_tower is None:
            raise RuntimeError("Model towers are not initialized. Call fit() or load_model() first.")
        user_embedding = self.user_tower(user_input)
        item_embedding = self.item_tower(item_input)
        return user_embedding, item_embedding

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        user_embedding, item_embedding = self.forward(user_input, item_input)
        user_embedding = F.normalize(user_embedding, dim=-1)
        item_embedding = F.normalize(item_embedding, dim=-1)
        return (user_embedding * item_embedding).sum(dim=-1)


class TwoTower(TwoTowerBase):
    def __init__(self, config: TwoTowerConfig | dict | None = None):
        if config is None:
            config = TwoTowerConfig()
        elif isinstance(config, dict):
            filtered_config = {
                field: value
                for field, value in config.items()
                if field in TwoTowerConfig.__dataclass_fields__
            }
            config = TwoTowerConfig(**filtered_config)

        super().__init__(config)
        self.device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.user_id_to_idx: dict[int, int] = {}
        self.item_id_to_idx: dict[int, int] = {}
        self.idx_to_user_id: list[int] = []
        self.idx_to_item_id: list[int] = []
        self.train_history: list[dict[str, float]] = []
        self.train_df: pd.DataFrame | None = None
        self.valid_df: pd.DataFrame | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> list[dict[str, float]]:
        self._fit_id_mappings(train_df, valid_df)
        self.train_df = self._prepare_interactions(train_df)
        self.valid_df = self._prepare_interactions(valid_df)
        self.build_towers(len(self.idx_to_user_id), len(self.idx_to_item_id))
        self.to(self.device)

        train_loader = self._make_loader(self.train_df, shuffle=True)
        valid_loader = self._make_loader(self.valid_df, shuffle=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        self.train_history = []
        for epoch in range(1, self.config.epochs + 1):
            self.train()
            train_loss_sum = 0.0
            train_examples = 0

            for user_batch, item_batch, label_batch in train_loader:
                user_batch = user_batch.to(self.device)
                item_batch = item_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.score_pairs(user_batch, item_batch)
                loss = criterion(logits, label_batch)
                loss.backward()
                optimizer.step()

                batch_size = label_batch.size(0)
                train_loss_sum += loss.item() * batch_size
                train_examples += batch_size

            metrics = self._evaluate_loader(valid_loader)
            metrics["epoch"] = float(epoch)
            metrics["train_loss"] = train_loss_sum / max(train_examples, 1)
            self.train_history.append(metrics)

            console.print(
                f"Epoch {epoch}/{self.config.epochs} "
                f"train_loss={metrics['train_loss']:.4f} "
                f"valid_loss={metrics['valid_loss']:.4f} "
                f"valid_accuracy={metrics['valid_accuracy']:.4f}"
            )

        return self.train_history

    def predict(
        self,
        user_ids: list[int] | None = None,
        item_ids: list[int] | None = None,
        top_k: int | None = None,
    ) -> dict[int, list[dict[str, float]]]:
        self._ensure_fitted()
        self.eval()

        top_k = top_k or self.config.top_k
        user_ids = user_ids or self.idx_to_user_id[: min(10, len(self.idx_to_user_id))]
        available_items = item_ids or self.idx_to_item_id

        item_indices = torch.tensor(
            [self.item_id_to_idx[item_id] for item_id in available_items if item_id in self.item_id_to_idx],
            dtype=torch.long,
            device=self.device,
        )
        if item_indices.numel() == 0:
            return {}

        with torch.no_grad():
            item_embeddings = self.item_tower(item_indices)
            item_embeddings = F.normalize(item_embeddings, dim=-1)

            predictions: dict[int, list[dict[str, float]]] = {}
            for user_id in user_ids:
                if user_id not in self.user_id_to_idx:
                    continue

                user_index = torch.tensor([self.user_id_to_idx[user_id]], dtype=torch.long, device=self.device)
                user_embedding = self.user_tower(user_index)
                user_embedding = F.normalize(user_embedding, dim=-1)
                scores = torch.matmul(item_embeddings, user_embedding.squeeze(0))
                k = min(top_k, scores.size(0))
                top_scores, top_indices = torch.topk(scores, k=k)

                predictions[user_id] = [
                    {
                        "banner_id": int(available_items[item_tensor_idx]),
                        "score": float(score),
                    }
                    for score, item_tensor_idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist())
                ]

        return predictions

    def evaluate(self, test_df: pd.DataFrame, top_k: int | None = None) -> dict[str, float]:
        self._ensure_fitted()
        prepared_test_df = self._prepare_interactions(test_df)
        if prepared_test_df.empty:
            raise RuntimeError("Test split is empty after filtering unknown users/items.")

        metrics = self._evaluate_loader(self._make_loader(prepared_test_df, shuffle=False), prefix="test")
        metrics["recall_at_k"] = self._recall_at_k(prepared_test_df, top_k or self.config.top_k)
        console.print(metrics)
        return metrics

    def save_model(self, path: str) -> None:
        self._ensure_fitted()
        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "config": asdict(self.config),
            "state_dict": self.state_dict(),
            "user_id_to_idx": self.user_id_to_idx,
            "item_id_to_idx": self.item_id_to_idx,
            "idx_to_user_id": self.idx_to_user_id,
            "idx_to_item_id": self.idx_to_item_id,
            "train_history": self.train_history,
        }
        torch.save(checkpoint, target_path)
        console.print(f"Model saved to {target_path}")

    def load_model(self, path: str) -> "TwoTower":
        checkpoint = torch.load(path, map_location=self.device)
        self.config = TwoTowerConfig(**checkpoint["config"])
        self.device = torch.device(self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.user_id_to_idx = checkpoint["user_id_to_idx"]
        self.item_id_to_idx = checkpoint["item_id_to_idx"]
        self.idx_to_user_id = checkpoint["idx_to_user_id"]
        self.idx_to_item_id = checkpoint["idx_to_item_id"]
        self.train_history = checkpoint.get("train_history", [])

        self.build_towers(len(self.idx_to_user_id), len(self.idx_to_item_id))
        self.load_state_dict(checkpoint["state_dict"])
        self.to(self.device)
        self.eval()
        return self

    def _fit_id_mappings(self, train_df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
        interactions = pd.concat([train_df, valid_df], ignore_index=True)
        self.idx_to_user_id = interactions["user_id"].astype(int).drop_duplicates().sort_values().tolist()
        self.idx_to_item_id = interactions["banner_id"].astype(int).drop_duplicates().sort_values().tolist()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.idx_to_user_id)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.idx_to_item_id)}

    def _prepare_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        required_columns = {"event_date", "user_id", "banner_id", "clicks"}
        missing_columns = required_columns.difference(interactions_df.columns)
        if missing_columns:
            raise ValueError(f"Interactions dataframe is missing columns: {sorted(missing_columns)}")

        interactions = interactions_df.loc[:, ["event_date", "user_id", "banner_id", "clicks"]].copy()
        interactions["event_date"] = pd.to_datetime(interactions["event_date"])
        interactions["user_id"] = interactions["user_id"].astype(int)
        interactions["banner_id"] = interactions["banner_id"].astype(int)
        interactions["label"] = (interactions["clicks"] > 0).astype("float32")
        interactions = interactions[
            interactions["user_id"].isin(self.user_id_to_idx)
            & interactions["banner_id"].isin(self.item_id_to_idx)
        ]

        if self.config.max_samples and len(interactions) > self.config.max_samples:
            positives = interactions[interactions["label"] == 1.0]
            negatives = interactions[interactions["label"] == 0.0]
            positive_target = min(len(positives), self.config.max_samples // 2)
            negative_target = min(len(negatives), self.config.max_samples - positive_target)

            sampled_frames = []
            if positive_target:
                sampled_frames.append(
                    positives.sample(n=positive_target, random_state=self.config.seed, replace=False)
                )
            if negative_target:
                sampled_frames.append(
                    negatives.sample(n=negative_target, random_state=self.config.seed, replace=False)
                )
            interactions = pd.concat(sampled_frames, ignore_index=True)

        return interactions.sort_values("event_date").reset_index(drop=True)

    def _make_loader(self, dataframe: pd.DataFrame, shuffle: bool) -> DataLoader:
        user_tensor = torch.tensor(
            dataframe["user_id"].map(self.user_id_to_idx).to_numpy(),
            dtype=torch.long,
        )
        item_tensor = torch.tensor(
            dataframe["banner_id"].map(self.item_id_to_idx).to_numpy(),
            dtype=torch.long,
        )
        label_tensor = torch.tensor(
            dataframe["label"].to_numpy(),
            dtype=torch.float32,
        )
        dataset = TensorDataset(user_tensor, item_tensor, label_tensor)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _evaluate_loader(self, loader: DataLoader, prefix: str = "valid") -> dict[str, float]:
        self.eval()
        criterion = nn.BCEWithLogitsLoss()
        loss_sum = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for user_batch, item_batch, label_batch in loader:
                user_batch = user_batch.to(self.device)
                item_batch = item_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                logits = self.score_pairs(user_batch, item_batch)
                loss = criterion(logits, label_batch)
                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).float()

                batch_size = label_batch.size(0)
                loss_sum += loss.item() * batch_size
                correct += (predictions == label_batch).sum().item()
                total += batch_size

        return {
            f"{prefix}_loss": loss_sum / max(total, 1),
            f"{prefix}_accuracy": correct / max(total, 1),
        }

    def _recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int) -> float:
        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        if positive_df.empty:
            return 0.0

        candidate_user_ids = (
            positive_df["user_id"]
            .drop_duplicates()
            .head(self.config.max_eval_users)
            .astype(int)
            .tolist()
        )
        recommendations = self.predict(user_ids=candidate_user_ids, top_k=top_k)

        recalls = []
        for user_id in candidate_user_ids:
            actual_items = set(positive_df.loc[positive_df["user_id"] == user_id, "banner_id"].astype(int))
            predicted_items = {
                row["banner_id"]
                for row in recommendations.get(user_id, [])
            }
            if actual_items:
                recalls.append(len(actual_items & predicted_items) / len(actual_items))

        return float(sum(recalls) / len(recalls)) if recalls else 0.0

    def _ensure_fitted(self) -> None:
        if self.user_tower is None or self.item_tower is None:
            raise RuntimeError("Model is not fitted yet.")
