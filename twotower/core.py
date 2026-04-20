from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Sequence, TypeAlias

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from twotower.config import TwoTowerConfig
from twotower.data import normalize_interactions
from twotower.evaluate import EvaluateInputs, TwoTowerEvaluator
from twotower.features import (
    FeatureMetadata,
    FeatureTables,
    build_item_feature_tables,
    build_user_feature_tables,
)
from twotower.fit import FitInputs, TwoTowerTrainer, build_pairwise_loader, compute_bpr_loss
from twotower.load_model import LoadedCheckpointState, TwoTowerModelLoader
from twotower.modules import TwoTowerBase
from twotower.predict import TwoTowerPredictor
from twotower.save_model import TwoTowerModelSaver

console = Console()

TargetLike: TypeAlias = pd.Series | Sequence[float]

class TwoTower(TwoTowerBase):
    def __init__(self, config: TwoTowerConfig | None = None):
        if config is None:
            config = TwoTowerConfig()

        super().__init__(config)
        self.config = config
        self.device = self.resolve_device(config.device)
        self.user_id_to_idx: dict[int, int] = {}
        self.item_id_to_idx: dict[int, int] = {}
        self.idx_to_user_id: list[int] = []
        self.idx_to_item_id: list[int] = []
        self.train_history: list[dict[str, float]] = []
        self.train_df: pd.DataFrame | None = None
        self.valid_df: pd.DataFrame | None = None
        self._seen_items_by_user: dict[int, set[int]] = {}
        self._train_positive_item_ids_by_popularity: list[int] = []
        self._cached_all_item_embeddings: torch.Tensor | None = None
        self._cached_all_item_ids: list[int] | None = None
        self._user_feature_tables: FeatureTables | None = None
        self._item_feature_tables: FeatureTables | None = None
        self._user_feature_metadata: FeatureMetadata = FeatureMetadata.empty()
        self._item_feature_metadata: FeatureMetadata = FeatureMetadata.empty()
        self._evaluator = TwoTowerEvaluator()
        self._predictor = TwoTowerPredictor()
        self._model_saver = TwoTowerModelSaver()
        self._model_loader = TwoTowerModelLoader()

    def fit(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: TargetLike,
        X_valid: pd.DataFrame,
        y_valid: TargetLike,
        users_df: pd.DataFrame | None = None,
        items_df: pd.DataFrame | None = None,
    ) -> list[dict[str, float]]:
        """Fit the model on interaction pairs.

        `X_train` and `X_valid` must contain `user_id` and `banner_id` columns.
        `y_train` and `y_valid` must contain the corresponding binary labels.
        """
        prepared_train_df, prepared_valid_df, reference_train_df, reference_valid_df = self._prepare_fit_inputs(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )
        self.train_df = reference_train_df
        self.valid_df = reference_valid_df
        self._refresh_evaluation_reference_data()
        self._prepare_side_feature_tables(users_df=users_df, items_df=items_df)

        self.invalidate_item_embedding_cache()

        fit_inputs = FitInputs(
            train_positive_df=prepared_train_df,
            valid_positive_df=prepared_valid_df,
            train_interactions_df=reference_train_df,
            valid_interactions_df=reference_valid_df,
            num_users=len(self.idx_to_user_id),
            num_items=len(self.idx_to_item_id),
        )
        trainer = TwoTowerTrainer(config=self.config, device=self.device)

        fit_result = trainer.fit(self, fit_inputs)
        self.train_history = fit_result.history

        self.invalidate_item_embedding_cache()

        return self.train_history

    def predict(
        self,
        user_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        exclude_seen: bool = True,
        strict: bool = False,
    ) -> dict[int, list[dict[str, float]]]:
        """Return top-k item recommendations for the requested users.

        If `user_ids` is omitted, predictions are generated for up to 10 known
        users. If `item_ids` is omitted, all known items are used as candidates.
        Unknown ids are skipped unless `strict=True`. Seen items are excluded by
        default.
        """
        self.ensure_fitted()
        return self._predictor.predict(
            self,
            user_ids=user_ids,
            item_ids=item_ids,
            top_k=top_k,
            exclude_seen=exclude_seen,
            strict=strict,
        )

    def evaluate(
        self,
        X_test: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on interaction pairs.

        `X_test` must contain `user_id` and `banner_id` columns. If a `clicks`
        column is present, labels are derived from it. If a `label` column is
        present, it is used directly.
        """
        metrics = self._evaluator.evaluate(
            self,
            X_test,
            top_k=top_k,
        )
        console.print(metrics)
        return metrics

    def save_model(self, path: str | PathLike[str]) -> None:
        """Save the fitted model checkpoint to disk."""
        target_path = self._model_saver.save_model(self, path)
        console.print(f"Model saved to {target_path}")

    def load_model(self, path: str | PathLike[str]) -> "TwoTower":
        """Load a model checkpoint from disk and return `self`."""
        self._model_loader.load_model(self, path)
        return self

    def validate_checkpoint(self, checkpoint: object, checkpoint_path: Path) -> None:
        """Validate checkpoint structure before loading model state."""
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format in {checkpoint_path}: expected a dictionary.")

        required_keys = {
            "config",
            "state_dict",
            "user_id_to_idx",
            "item_id_to_idx",
            "idx_to_user_id",
            "idx_to_item_id",
        }
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise ValueError(
                f"Invalid checkpoint format in {checkpoint_path}: missing keys {sorted(missing_keys)}."
            )

    def apply_loaded_checkpoint_state(self, state: LoadedCheckpointState) -> None:
        self.config = state.config
        self.device = state.device
        self.user_id_to_idx = state.user_id_to_idx
        self.item_id_to_idx = state.item_id_to_idx
        self.idx_to_user_id = state.idx_to_user_id
        self.idx_to_item_id = state.idx_to_item_id
        self.train_history = state.train_history
        self.train_df = None
        self.valid_df = None
        self._seen_items_by_user = state.seen_items_by_user
        self._train_positive_item_ids_by_popularity = state.train_positive_item_ids_by_popularity
        self._user_feature_tables = None
        self._item_feature_tables = None
        self._user_feature_metadata = state.user_feature_metadata
        self._item_feature_metadata = state.item_feature_metadata

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs:
        test_input_df = self._prepare_evaluation_inputs(X_test)
        prepared_test_df = self._prepare_interactions(test_input_df, apply_sampling=False)
        positive_test_df = (
            prepared_test_df.copy()
            if prepared_test_df.empty
            else self._prepare_retrieval_pairs(
                prepared_test_df,
                apply_sampling=False,
                split_name="test",
            )
        )
        return EvaluateInputs(
            test_input_df=test_input_df,
            prepared_test_df=prepared_test_df,
            positive_test_df=positive_test_df,
            input_row_count=len(test_input_df),
            unknown_user_row_count=int((~test_input_df["user_id"].isin(self.user_id_to_idx)).sum()),
            unknown_item_row_count=int((~test_input_df["banner_id"].isin(self.item_id_to_idx)).sum()),
        )

    def make_loader(
        self,
        *,
        positive_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader:
        return build_pairwise_loader(
            positive_df=positive_df,
            interactions_df=interactions_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            num_items=len(self.idx_to_item_id),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            observed_negative_sampling_ratio=self.config.observed_negative_sampling_ratio,
            seed=self.config.seed + 2,
        )

    def evaluate_loader(self, loader: DataLoader, prefix: str = "valid") -> dict[str, float]:
        self.eval()
        criterion = nn.LogSigmoid()
        loss_sum = 0.0
        total = 0

        with torch.no_grad():
            for user_batch, pos_item_batch, neg_item_batch in loader:
                user_batch = user_batch.to(self.device)
                pos_item_batch = pos_item_batch.to(self.device)
                neg_item_batch = neg_item_batch.to(self.device)

                positive_scores = self.score_pairs(user_batch, pos_item_batch)
                negative_scores = self.score_pairs(user_batch, neg_item_batch)
                loss = compute_bpr_loss(
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                    criterion=criterion,
                )

                batch_size = user_batch.size(0)
                loss_sum += loss.item() * batch_size
                total += batch_size

        return {
            f"{prefix}_loss": loss_sum / max(total, 1),
        }

    def resolve_eval_top_ks(self, top_k: int | None) -> list[int]:
        requested_top_ks = list(self.config.eval_top_ks)
        requested_top_ks.append(top_k or self.config.top_k)

        resolved_top_ks: list[int] = []
        for candidate_top_k in requested_top_ks:
            candidate_value = int(candidate_top_k)
            if candidate_value <= 0:
                raise ValueError("Evaluation top-k values must be positive integers.")
            if candidate_value not in resolved_top_ks:
                resolved_top_ks.append(candidate_value)
        return resolved_top_ks

    def get_eval_user_ids(self, evaluation_df: pd.DataFrame) -> list[int]:
        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        if positive_df.empty:
            return []

        return (
            positive_df["user_id"]
            .drop_duplicates()
            .head(self.config.max_eval_users)
            .astype(int)
            .tolist()
        )

    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int) -> float:
        candidate_user_ids = self.get_eval_user_ids(evaluation_df)
        if not candidate_user_ids:
            return 0.0

        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        recalls = []
        seen_items_by_user = self.get_seen_items_by_user()
        item_embeddings, item_ids = self.get_candidate_item_embeddings(list(self.idx_to_item_id))
        for user_id in candidate_user_ids:
            actual_items = set(positive_df.loc[positive_df["user_id"] == user_id, "banner_id"].astype(int))
            predicted_items = self._predictor.predict_top_k_item_ids_for_user(
                self,
                user_id=user_id,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
                top_k=top_k,
                excluded_item_ids=seen_items_by_user.get(user_id, set()),
            )
            if actual_items:
                recalls.append(len(actual_items & predicted_items) / len(actual_items))

        return float(sum(recalls) / len(recalls)) if recalls else 0.0

    def popularity_recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int) -> float:
        candidate_user_ids = self.get_eval_user_ids(evaluation_df)
        if not candidate_user_ids:
            return 0.0

        popularity_ranking = self.get_train_positive_item_ranking()
        if not popularity_ranking:
            return 0.0

        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        recalls = []
        seen_items_by_user = self.get_seen_items_by_user()
        for user_id in candidate_user_ids:
            actual_items = set(positive_df.loc[positive_df["user_id"] == user_id, "banner_id"].astype(int))
            if not actual_items:
                continue

            excluded_item_ids = seen_items_by_user.get(user_id, set())
            predicted_items: list[int] = []
            for item_id in popularity_ranking:
                if item_id in excluded_item_ids:
                    continue
                predicted_items.append(item_id)
                if len(predicted_items) == top_k:
                    break

            recalls.append(len(actual_items & set(predicted_items)) / len(actual_items))

        return float(sum(recalls) / len(recalls)) if recalls else 0.0

    def ensure_fitted(self) -> None:
        if self.user_tower is None or self.item_tower is None:
            raise RuntimeError("Model is not fitted yet.")

    def get_seen_items_by_user(self) -> dict[int, set[int]]:
        if self._seen_items_by_user:
            return self._seen_items_by_user

        self._refresh_evaluation_reference_data()
        return self._seen_items_by_user

    def get_train_positive_item_ranking(self) -> list[int]:
        if self._train_positive_item_ids_by_popularity:
            return self._train_positive_item_ids_by_popularity

        self._refresh_evaluation_reference_data()
        return self._train_positive_item_ids_by_popularity

    def get_candidate_item_embeddings(
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

    def get_user_embedding(self, user_id: int) -> torch.Tensor:
        if user_id not in self.user_id_to_idx:
            raise KeyError(f"Unknown user_id: {user_id}")

        user_index = torch.tensor(
            [self.user_id_to_idx[user_id]],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            user_embedding = self.user_tower(user_index)
        return user_embedding.squeeze(0)

    def get_user_feature_metadata_dict(self) -> dict[str, object]:
        return self._user_feature_metadata.to_dict()

    def get_item_feature_metadata_dict(self) -> dict[str, object]:
        return self._item_feature_metadata.to_dict()

    def invalidate_item_embedding_cache(self) -> None:
        self._cached_all_item_embeddings = None
        self._cached_all_item_ids = None

    @staticmethod
    def resolve_device(device: str | None) -> torch.device:
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device)

    def _fit_id_mappings(self, train_df: pd.DataFrame) -> None:
        self.idx_to_user_id = train_df["user_id"].astype(int).drop_duplicates().sort_values().tolist()
        self.idx_to_item_id = train_df["banner_id"].astype(int).drop_duplicates().sort_values().tolist()
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(self.idx_to_user_id)}
        self.item_id_to_idx = {item_id: idx for idx, item_id in enumerate(self.idx_to_item_id)}

    def _prepare_side_feature_tables(
        self,
        *,
        users_df: pd.DataFrame | None,
        items_df: pd.DataFrame | None,
    ) -> None:
        if users_df is None and items_df is None:
            self._user_feature_tables = None
            self._item_feature_tables = None
            self._user_feature_metadata = FeatureMetadata.empty()
            self._item_feature_metadata = FeatureMetadata.empty()
            return

        if users_df is None or items_df is None:
            raise ValueError("`users_df` and `items_df` must be provided together when using side features.")

        self._user_feature_tables = build_user_feature_tables(
            users_df=users_df,
            user_ids=self.idx_to_user_id,
        )
        self._item_feature_tables = build_item_feature_tables(
            items_df=items_df,
            item_ids=self.idx_to_item_id,
        )
        self._user_feature_metadata = self._user_feature_tables.metadata
        self._item_feature_metadata = self._item_feature_tables.metadata

    def _prepare_interactions(
        self,
        interactions_df: pd.DataFrame,
        apply_sampling: bool = False,
    ) -> pd.DataFrame:
        if "label" in interactions_df.columns:
            prepared_interactions = interactions_df.copy()
            if "event_date" not in prepared_interactions.columns:
                prepared_interactions["event_date"] = pd.Timestamp("1970-01-01")
            return self._filter_and_sample_interactions(
                interactions_df=prepared_interactions,
                apply_sampling=apply_sampling,
                sort_by_event_date=True,
            )

        normalized_interactions = normalize_interactions(interactions_df)
        return self._filter_and_sample_interactions(
            interactions_df=normalized_interactions,
            apply_sampling=apply_sampling,
            sort_by_event_date=True,
        )

    def _prepare_evaluation_inputs(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalize evaluation inputs to interaction format."""
        if not isinstance(X_test, pd.DataFrame):
            raise TypeError(
                "Evaluation features must be a pandas DataFrame with "
                "'user_id' and 'banner_id' columns."
            )

        required_columns = {"user_id", "banner_id"}
        missing_columns = required_columns.difference(X_test.columns)
        if missing_columns:
            raise ValueError(
                f"Evaluation features are missing required columns: {sorted(missing_columns)}"
            )

        if "label" in X_test.columns:
            evaluation_df = X_test.copy()
            evaluation_df["user_id"] = evaluation_df["user_id"].astype(int)
            evaluation_df["banner_id"] = evaluation_df["banner_id"].astype(int)
            evaluation_df["label"] = evaluation_df["label"].astype("float32")
            if "event_date" not in evaluation_df.columns:
                evaluation_df["event_date"] = pd.Timestamp("1970-01-01")
            else:
                evaluation_df["event_date"] = pd.to_datetime(evaluation_df["event_date"])
            return evaluation_df.loc[:, ["event_date", "user_id", "banner_id", "label"]]

        if "clicks" not in X_test.columns:
            raise ValueError(
                "Evaluation features must include either a 'label' column or a 'clicks' column."
            )

        evaluation_df = X_test.copy()
        if "event_date" not in evaluation_df.columns:
            evaluation_df["event_date"] = pd.Timestamp("1970-01-01")
        return normalize_interactions(evaluation_df)

    def _prepare_fit_inputs(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: TargetLike,
        X_valid: pd.DataFrame,
        y_valid: TargetLike,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Validate fit inputs, build id mappings, and prepare train/valid splits."""
        train_df = self._build_labeled_interactions(X_train, y_train, split_name="train")
        valid_df = self._build_labeled_interactions(X_valid, y_valid, split_name="valid")
        self._fit_id_mappings(train_df)

        prepared_train_df = self._prepare_retrieval_pairs(
            train_df,
            apply_sampling=True,
            split_name="train",
        )
        prepared_valid_df = self._prepare_retrieval_pairs(
            valid_df,
            apply_sampling=False,
            split_name="valid",
        )
        reference_train_df = self._filter_and_sample_interactions(
            interactions_df=train_df,
            apply_sampling=False,
            sort_by_event_date=False,
        )
        reference_valid_df = self._filter_and_sample_interactions(
            interactions_df=valid_df,
            apply_sampling=False,
            sort_by_event_date=False,
        )
        return prepared_train_df, prepared_valid_df, reference_train_df, reference_valid_df

    def _prepare_retrieval_pairs(
        self,
        interactions_df: pd.DataFrame,
        apply_sampling: bool,
        split_name: str,
    ) -> pd.DataFrame:
        filtered_interactions = self._filter_and_sample_interactions(
            interactions_df=interactions_df,
            apply_sampling=False,
            sort_by_event_date=False,
        )
        positive_interactions = filtered_interactions[filtered_interactions["label"] == 1.0].copy()
        if positive_interactions.empty:
            raise ValueError(f"{split_name} split has no positive interactions for retrieval training.")

        if apply_sampling and self.config.max_samples and len(positive_interactions) > self.config.max_samples:
            positive_interactions = positive_interactions.sample(
                n=self.config.max_samples,
                random_state=self.config.seed,
                replace=False,
            )

        return positive_interactions.reset_index(drop=True)

    def _build_labeled_interactions(
        self,
        X: pd.DataFrame,
        y: TargetLike,
        split_name: str,
    ) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"{split_name} features must be a pandas DataFrame with "
                "'user_id' and 'banner_id' columns."
            )
        required_columns = {"user_id", "banner_id"}
        missing_columns = required_columns.difference(X.columns)
        if missing_columns:
            raise ValueError(
                f"{split_name} features are missing required columns: {sorted(missing_columns)}"
            )

        y_series = y if isinstance(y, pd.Series) else pd.Series(y, name="label")
        if len(X) != len(y_series):
            raise ValueError(
                f"{split_name} features and labels must have the same length: "
                f"{len(X)} != {len(y_series)}"
            )

        prepared_df = X.loc[:, ["user_id", "banner_id"]].copy()
        prepared_df["label"] = y_series.to_numpy()
        return prepared_df.reset_index(drop=True)

    def _filter_and_sample_interactions(
        self,
        interactions_df: pd.DataFrame,
        apply_sampling: bool = False,
        sort_by_event_date: bool = False,
    ) -> pd.DataFrame:
        required_columns = {"user_id", "banner_id", "label"}
        missing_columns = required_columns.difference(interactions_df.columns)
        if missing_columns:
            raise ValueError(
                f"Prepared interactions dataframe is missing columns: {sorted(missing_columns)}"
            )

        selected_columns = ["user_id", "banner_id", "label"]
        if "event_date" in interactions_df.columns:
            selected_columns.append("event_date")

        interactions = interactions_df.loc[:, selected_columns].copy()
        interactions["user_id"] = interactions["user_id"].astype(int)
        interactions["banner_id"] = interactions["banner_id"].astype(int)
        interactions["label"] = interactions["label"].astype("float32")
        interactions = interactions[
            interactions["user_id"].isin(self.user_id_to_idx)
            & interactions["banner_id"].isin(self.item_id_to_idx)
        ]

        if apply_sampling and self.config.max_samples and len(interactions) > self.config.max_samples:
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

        if sort_by_event_date and "event_date" in interactions.columns:
            interactions = interactions.sort_values("event_date")

        return interactions.reset_index(drop=True)

    def _refresh_evaluation_reference_data(self) -> None:
        self._seen_items_by_user = {}
        for dataframe in (self.train_df, self.valid_df):
            if dataframe is None or dataframe.empty:
                continue
            grouped = dataframe.groupby("user_id")["banner_id"]
            for user_id, item_ids in grouped:
                self._seen_items_by_user.setdefault(int(user_id), set()).update(
                    int(item_id) for item_id in item_ids.tolist()
                )

        if self.train_df is None or self.train_df.empty:
            self._train_positive_item_ids_by_popularity = []
            return

        self._train_positive_item_ids_by_popularity = (
            self.train_df.loc[self.train_df["label"] == 1.0, "banner_id"]
            .astype(int)
            .value_counts()
            .index
            .tolist()
        )

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
