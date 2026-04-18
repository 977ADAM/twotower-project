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
from twotower.data import normalize_interactions, prepare_interactions, split_interactions
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
    def __init__(self, config: TwoTowerConfig | None = None):
        if config is None:
            config = TwoTowerConfig()

        super().__init__(config)
        self.config = config
        self.device = self._resolve_device(config.device)
        self.user_id_to_idx: dict[int, int] = {}
        self.item_id_to_idx: dict[int, int] = {}
        self.idx_to_user_id: list[int] = []
        self.idx_to_item_id: list[int] = []
        self.train_history: list[dict[str, float]] = []
        self.train_df: pd.DataFrame | None = None
        self.valid_df: pd.DataFrame | None = None
        self._cached_all_item_embeddings: torch.Tensor | None = None
        self._cached_all_item_ids: list[int] | None = None

    def fit(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
    ) -> list[dict[str, float]]:
        self._fit_id_mappings(train_df)
        self.train_df = self._prepare_interactions(train_df, apply_sampling=True)
        self.valid_df = self._prepare_interactions(valid_df, apply_sampling=False)
        self.build_towers(len(self.idx_to_user_id), len(self.idx_to_item_id))
        self.to(self.device)
        self._invalidate_item_embedding_cache()

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

    def fit_from_interactions(
        self,
        interactions_df: pd.DataFrame,
    ) -> tuple[list[dict[str, float]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        normalized_interactions = normalize_interactions(interactions_df)
        train_df, valid_df, test_df = split_interactions(
            normalized_interactions,
            validation_ratio=self.config.validation_ratio,
            test_ratio=self.config.test_ratio,
        )
        history = self.fit(train_df, valid_df)
        prepared_test_df = self._prepare_interactions(test_df, apply_sampling=False)
        return history, self.train_df, self.valid_df, prepared_test_df

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
        candidate_items = item_ids or self.idx_to_item_id
        available_items = [
            item_id for item_id in candidate_items if item_id in self.item_id_to_idx
        ]
        if not available_items:
            return {}

        item_embeddings, item_ids = self._get_item_embeddings_for_candidates(available_items)

        with torch.no_grad():
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
                        "banner_id": int(item_ids[item_tensor_idx]),
                        "score": float(score),
                    }
                    for score, item_tensor_idx in zip(top_scores.cpu().tolist(), top_indices.cpu().tolist())
                ]

        return predictions

    def evaluate(self, test_df: pd.DataFrame, top_k: int | None = None) -> dict[str, float]:
        self._ensure_fitted()
        required_columns = {"user_id", "banner_id"}
        missing_columns = required_columns.difference(test_df.columns)
        if missing_columns:
            raise ValueError(f"Test dataframe is missing columns: {sorted(missing_columns)}")
        test_input_rows = len(test_df)
        test_unknown_user_rows = int((~test_df["user_id"].isin(self.user_id_to_idx)).sum())
        test_unknown_item_rows = int((~test_df["banner_id"].isin(self.item_id_to_idx)).sum())
        prepared_test_df = self._prepare_interactions(test_df, apply_sampling=False)
        if prepared_test_df.empty:
            raise RuntimeError("Test split is empty after filtering unknown users/items.")

        metrics = self._evaluate_loader(self._make_loader(prepared_test_df, shuffle=False), prefix="test")
        metrics["recall_at_k"] = self._recall_at_k(prepared_test_df, top_k or self.config.top_k)
        metrics["test_input_rows"] = float(test_input_rows)
        metrics["test_rows_used"] = float(len(prepared_test_df))
        metrics["test_rows_filtered"] = float(test_input_rows - len(prepared_test_df))
        metrics["test_unknown_user_rows"] = float(test_unknown_user_rows)
        metrics["test_unknown_item_rows"] = float(test_unknown_item_rows)
        metrics["test_rows_filtered_ratio"] = (
            float(test_input_rows - len(prepared_test_df)) / max(float(test_input_rows), 1.0)
        )
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
        checkpoint = torch.load(path, map_location="cpu")
        self.config = TwoTowerConfig(**checkpoint["config"])
        self.device = self._resolve_device(self.config.device)
        self.user_id_to_idx = checkpoint["user_id_to_idx"]
        self.item_id_to_idx = checkpoint["item_id_to_idx"]
        self.idx_to_user_id = checkpoint["idx_to_user_id"]
        self.idx_to_item_id = checkpoint["idx_to_item_id"]
        self.train_history = checkpoint.get("train_history", [])

        self.build_towers(len(self.idx_to_user_id), len(self.idx_to_item_id))
        self.load_state_dict(checkpoint["state_dict"])
        self.to(self.device)
        self._invalidate_item_embedding_cache()
        self.eval()
        return self

    def _fit_id_mappings(self, train_df: pd.DataFrame) -> None:
        self.idx_to_user_id = train_df["user_id"].astype(int).drop_duplicates().sort_values().tolist()
        self.idx_to_item_id = train_df["banner_id"].astype(int).drop_duplicates().sort_values().tolist()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.idx_to_user_id)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.idx_to_item_id)}

    def _prepare_interactions(
        self,
        interactions_df: pd.DataFrame,
        apply_sampling: bool = False,
    ) -> pd.DataFrame:
        return prepare_interactions(
            interactions_df=interactions_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            max_samples=self.config.max_samples if apply_sampling else None,
            seed=self.config.seed,
        )

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

        recalls = []
        seen_items_by_user = self._build_seen_items_by_user()
        item_embeddings, item_ids = self._build_candidate_item_embeddings()
        for user_id in candidate_user_ids:
            actual_items = set(positive_df.loc[positive_df["user_id"] == user_id, "banner_id"].astype(int))
            predicted_items = self._predict_top_k_for_user(
                user_id=user_id,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
                top_k=top_k,
                excluded_item_ids=seen_items_by_user.get(user_id, set()),
            )
            if actual_items:
                recalls.append(len(actual_items & predicted_items) / len(actual_items))

        return float(sum(recalls) / len(recalls)) if recalls else 0.0

    def _ensure_fitted(self) -> None:
        if self.user_tower is None or self.item_tower is None:
            raise RuntimeError("Model is not fitted yet.")

    def _build_seen_items_by_user(self) -> dict[int, set[int]]:
        seen_items_by_user: dict[int, set[int]] = {}
        for dataframe in (self.train_df, self.valid_df):
            if dataframe is None or dataframe.empty:
                continue
            grouped = dataframe.groupby("user_id")["banner_id"]
            for user_id, item_ids in grouped:
                seen_items_by_user.setdefault(int(user_id), set()).update(
                    int(item_id) for item_id in item_ids.tolist()
                )
        return seen_items_by_user

    def _build_candidate_item_embeddings(self) -> tuple[torch.Tensor, list[int]]:
        if self._cached_all_item_embeddings is None or self._cached_all_item_ids is None:
            item_ids = list(self.idx_to_item_id)
            item_indices = torch.tensor(
                [self.item_id_to_idx[item_id] for item_id in item_ids],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                item_embeddings = self.item_tower(item_indices)
                item_embeddings = F.normalize(item_embeddings, dim=-1)
            self._cached_all_item_embeddings = item_embeddings
            self._cached_all_item_ids = item_ids
        return self._cached_all_item_embeddings, self._cached_all_item_ids

    def _get_item_embeddings_for_candidates(
        self,
        item_ids: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        all_item_embeddings, all_item_ids = self._build_candidate_item_embeddings()
        if item_ids == all_item_ids:
            return all_item_embeddings, all_item_ids

        candidate_positions = [
            self.item_id_to_idx[item_id]
            for item_id in item_ids
            if item_id in self.item_id_to_idx
        ]
        if not candidate_positions:
            return all_item_embeddings[:0], []

        return all_item_embeddings[candidate_positions], item_ids

    def _predict_top_k_for_user(
        self,
        user_id: int,
        item_embeddings: torch.Tensor,
        item_ids: list[int],
        top_k: int,
        excluded_item_ids: set[int] | None = None,
    ) -> set[int]:
        if user_id not in self.user_id_to_idx:
            return set()

        excluded_item_ids = excluded_item_ids or set()
        candidate_positions = [
            position for position, item_id in enumerate(item_ids)
            if item_id not in excluded_item_ids
        ]
        if not candidate_positions:
            return set()

        user_index = torch.tensor([self.user_id_to_idx[user_id]], dtype=torch.long, device=self.device)
        with torch.no_grad():
            user_embedding = self.user_tower(user_index)
            user_embedding = F.normalize(user_embedding, dim=-1)
            candidate_embeddings = item_embeddings[candidate_positions]
            scores = torch.matmul(candidate_embeddings, user_embedding.squeeze(0))

        k = min(top_k, scores.size(0))
        top_positions = torch.topk(scores, k=k).indices.cpu().tolist()
        return {
            int(item_ids[candidate_positions[position]])
            for position in top_positions
        }

    def _invalidate_item_embedding_cache(self) -> None:
        self._cached_all_item_embeddings = None
        self._cached_all_item_ids = None

    @staticmethod
    def _resolve_device(device: str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)
