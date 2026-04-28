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

from twotower._src.config import TwoTowerConfig
from twotower._src.evaluate import EvaluateInputs, TwoTowerEvaluator
from twotower._src.features import (
    FeatureConfig,
    FeatureMetadata,
    FeatureTables,
    build_feature_tables,
)
from twotower._src.modules import ItemTower, UserTower
from twotower._src.fit import EarlyStopping, FitInputs, NegativeSampling, TwoTowerTrainer, build_pairwise_loader, compute_bpr_loss
from twotower._src.load_model import LoadedCheckpointState, TwoTowerModelLoader
from twotower._src.modules import TwoTowerBase
from twotower._src.predict import TwoTowerPredictor
from twotower._src.preprocessing import (
    build_evaluation_reference_data,
    build_id_mappings,
    build_labeled_interactions,
    filter_and_sample_interactions,
    normalize_and_filter_interactions,
    prepare_evaluation_inputs,
    prepare_retrieval_pairs,
)
from twotower._src.save_model import TwoTowerModelSaver
from twotower._src.utils.traceback_utils import filter_traceback

console = Console()

TargetLike: TypeAlias = pd.Series | Sequence[float]


class TwoTower(TwoTowerBase):
    """Two-tower retrieval model with a scikit-learn–style API.

    Both towers share the same MLP architecture and embed users and items
    into a common `hidden_dim`-dimensional space. Similarity is measured
    with a dot product. Training minimises BPR loss with optional in-batch
    InfoNCE contrastive loss.

    Args:
        user_col: Column name for user IDs in the interaction DataFrames.
        item_col: Column name for item IDs in the interaction DataFrames.
        user_embedding_dim: Dimensionality of the learned user ID embedding.
        item_embedding_dim: Dimensionality of the learned item ID embedding.
        side_feature_embedding_dim: Dimensionality of each side-feature embedding.
        hidden_dim: Output embedding size for both towers.
        tower_dims: Hidden layer sizes for the MLP inside each tower.
            Each layer is followed by BatchNorm1d, ReLU, and optional Dropout.
            Pass an empty tuple ``()`` for a single linear projection.
        dropout: Dropout rate applied after each hidden layer in the MLP.
            Must be in ``[0, 1)``.
        retrieval_temperature: Temperature for the in-batch InfoNCE loss.
        learning_rate: Adam optimizer learning rate.
        batch_size: Mini-batch size for training.
        epochs: Maximum number of training epochs.
        max_samples: Cap on the number of positive training pairs per epoch.
            ``None`` uses all available pairs.
        eval_top_ks: Top-k values used when computing recall metrics.
        max_eval_users: Maximum number of users sampled for evaluation.
        top_k: Default number of recommendations returned by ``predict``.
        eval_during_training: Whether to compute recall metrics after each epoch.
        seed: Random seed for reproducibility.
        device: PyTorch device string (``"cpu"``, ``"cuda"``), or ``None`` to
            auto-detect.

    Example:
        ```python
        from twotower import TwoTower, split_interactions

        train_df, valid_df, test_df = split_interactions(interactions_df)

        model = TwoTower(epochs=10, tower_dims=(256, 128))
        model.fit(
            X_train=train_df.drop(columns=["clicks"]),
            y_train=train_df["clicks"],
            X_valid=valid_df.drop(columns=["clicks"]),
            y_valid=valid_df["clicks"],
        )
        recommendations = model.predict(user_ids=[1, 2, 3], top_k=10)
        metrics = model.evaluate(test_df)
        model.save_model("model.pth")
        ```
    """

    def __init__(
        self,
        *,
        user_col: str = "user_id",
        item_col: str = "banner_id",
        user_embedding_dim: int = 64,
        item_embedding_dim: int = 64,
        side_feature_embedding_dim: int = 8,
        hidden_dim: int = 64,
        tower_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.0,
        retrieval_temperature: float = 0.1,
        learning_rate: float = 1e-3,
        batch_size: int = 2048,
        epochs: int = 25,
        max_samples: int | None = 250_000,
        eval_top_ks: tuple[int, ...] = (50, 100, 300),
        max_eval_users: int = 500,
        top_k: int = 100,
        eval_during_training: bool = True,
        seed: int = 42,
        device: str | None = "cpu",
    ):
        config = TwoTowerConfig(
            user_embedding_dim=user_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            side_feature_embedding_dim=side_feature_embedding_dim,
            hidden_dim=hidden_dim,
            tower_dims=tower_dims,
            dropout=dropout,
            retrieval_temperature=retrieval_temperature,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            max_samples=max_samples,
            eval_top_ks=eval_top_ks,
            max_eval_users=max_eval_users,
            top_k=top_k,
            eval_during_training=eval_during_training,
            seed=seed,
            device=device,
        )
        super().__init__(config)
        self.config = config
        self.user_col = user_col
        self.item_col = item_col
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
        self._negative_sampling: NegativeSampling = NegativeSampling()
        self._evaluator = TwoTowerEvaluator()
        self._predictor = TwoTowerPredictor()
        self._model_saver = TwoTowerModelSaver()
        self._model_loader = TwoTowerModelLoader()

    @filter_traceback
    def fit(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: TargetLike,
        X_valid: pd.DataFrame,
        y_valid: TargetLike,
        users_df: pd.DataFrame | None = None,
        items_df: pd.DataFrame | None = None,
        user_feature_config: FeatureConfig | None = None,
        item_feature_config: FeatureConfig | None = None,
        negative_sampling: NegativeSampling = NegativeSampling(),
        early_stopping: EarlyStopping | None = EarlyStopping(),
    ) -> list[dict[str, float]]:
        """Fit the model on interaction pairs.

        Args:
            X_train: Training interactions. Must contain the columns named by
                ``user_col`` and ``item_col``.
            y_train: Binary labels for ``X_train`` (positive = 1, negative = 0).
                May be a pandas Series or any sequence of floats.
            X_valid: Validation interactions in the same format as ``X_train``.
            y_valid: Binary labels for ``X_valid``.
            users_df: Side-feature table for users. Must be provided together
                with ``items_df`` and the feature config arguments.
            items_df: Side-feature table for items.
            user_feature_config: Declares which columns in ``users_df`` to encode
                as side features.
            item_feature_config: Declares which columns in ``items_df`` to encode
                as side features.
            negative_sampling: Strategy for drawing negative examples.
            early_stopping: Early stopping configuration. Pass ``None`` to train
                for the full number of epochs.

        Returns:
            A list of per-epoch metric dicts (train loss, valid loss, recall@k, …).
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
        self._prepare_side_feature_tables(
            users_df=users_df,
            items_df=items_df,
            user_feature_config=user_feature_config,
            item_feature_config=item_feature_config,
        )
        self._negative_sampling = negative_sampling

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

        fit_result = trainer.fit(
            self, fit_inputs,
            negative_sampling=negative_sampling,
            early_stopping=early_stopping,
        )
        self.train_history = fit_result.history

        self.invalidate_item_embedding_cache()

        return self.train_history

    @filter_traceback
    def predict(
        self,
        user_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        exclude_seen: bool = True,
        strict: bool = False,
    ) -> dict[int, list[dict[str, float]]]:
        """Return top-k item recommendations for the requested users.

        Args:
            user_ids: User IDs to generate recommendations for. If ``None``,
                up to 10 known users are used.
            item_ids: Candidate item IDs. If ``None``, all known items are used.
            top_k: Number of items to return per user. Defaults to the model's
                ``top_k`` config value.
            exclude_seen: Whether to exclude items the user interacted with
                during training.
            strict: If ``True``, raises ``KeyError`` for unknown user or item IDs.
                If ``False``, they are silently skipped.

        Returns:
            A dict mapping each user ID to a ranked list of
            ``{item_col: item_id, "score": float}`` dicts.
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

    @filter_traceback
    def evaluate(
        self,
        X_test: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, float]:
        """Evaluate the model on a held-out test set.

        Args:
            X_test: Test interactions. Must contain the user and item ID columns.
                Labels are derived from a ``clicks`` column (``clicks > 0`` → 1)
                or a ``label`` column if present.
            top_k: Additional top-k value to evaluate at, on top of
                ``eval_top_ks``.

        Returns:
            A dict of recall@k and popularity_recall@k metrics.
        """
        metrics = self._evaluator.evaluate(self, X_test, top_k=top_k)
        console.print(metrics)
        return metrics

    @filter_traceback
    def save_model(self, path: str | PathLike[str]) -> None:
        """Save the fitted model checkpoint to disk."""
        target_path = self._model_saver.save_model(self, path)
        console.print(f"Model saved to {target_path}")

    @filter_traceback
    def load_model(self, path: str | PathLike[str]) -> "TwoTower":
        """Load a model checkpoint from disk and return `self`."""
        self._model_loader.load_model(self, path)
        return self

    # ── Protocol methods required by service modules ──────────────────────────

    def validate_checkpoint(self, checkpoint: object, checkpoint_path: Path) -> None:
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint format in {checkpoint_path}: expected a dictionary.")

        required_keys = {"config", "state_dict", "user_id_to_idx", "item_id_to_idx", "idx_to_user_id", "idx_to_item_id"}
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise ValueError(
                f"Invalid checkpoint format in {checkpoint_path}: missing keys {sorted(missing_keys)}."
            )

    def apply_loaded_checkpoint_state(self, state: LoadedCheckpointState) -> None:
        self.config = state.config
        self.user_col = state.user_col
        self.item_col = state.item_col
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
        test_input_df = prepare_evaluation_inputs(X_test, self.user_col, self.item_col)
        prepared_test_df = normalize_and_filter_interactions(
            test_input_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            config=self.config,
            apply_sampling=False,
        )
        positive_test_df = (
            prepared_test_df.copy()
            if prepared_test_df.empty
            else prepare_retrieval_pairs(
                prepared_test_df,
                user_id_to_idx=self.user_id_to_idx,
                item_id_to_idx=self.item_id_to_idx,
                config=self.config,
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
            unknown_item_row_count=int((~test_input_df["banner_id"].isin(self.item_id_to_idx)).sum()),  # internal name after boundary rename
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
            observed_negative_sampling_ratio=self._negative_sampling.observed_ratio,
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

        return {f"{prefix}_loss": loss_sum / max(total, 1)}

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

    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float:
        candidate_user_ids = self.get_eval_user_ids(evaluation_df)
        if not candidate_user_ids:
            return 0.0

        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        recalls = []
        seen_items_by_user = self.get_seen_items_by_user() if exclude_seen else {}
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

    def build_towers(self, num_users: int, num_items: int) -> None:
        self.user_tower = UserTower(
            num_users,
            self.config,
            feature_tables=self._user_feature_tables,
            feature_metadata=self._user_feature_metadata,
        )
        self.item_tower = ItemTower(
            num_items,
            self.config,
            feature_tables=self._item_feature_tables,
            feature_metadata=self._item_feature_metadata,
        )

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

    # ── Private orchestration helpers ─────────────────────────────────────────

    def _prepare_fit_inputs(
        self,
        *,
        X_train: pd.DataFrame,
        y_train: TargetLike,
        X_valid: pd.DataFrame,
        y_valid: TargetLike,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train_df = build_labeled_interactions(X_train, y_train, split_name="train", user_col=self.user_col, item_col=self.item_col)
        valid_df = build_labeled_interactions(X_valid, y_valid, split_name="valid", user_col=self.user_col, item_col=self.item_col)

        mappings = build_id_mappings(train_df)
        self.user_id_to_idx = mappings.user_id_to_idx
        self.item_id_to_idx = mappings.item_id_to_idx
        self.idx_to_user_id = mappings.idx_to_user_id
        self.idx_to_item_id = mappings.idx_to_item_id

        prepared_train_df = prepare_retrieval_pairs(
            train_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            config=self.config,
            apply_sampling=True,
            split_name="train",
        )
        prepared_valid_df = prepare_retrieval_pairs(
            valid_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            config=self.config,
            apply_sampling=False,
            split_name="valid",
        )
        reference_train_df = filter_and_sample_interactions(
            train_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            config=self.config,
            apply_sampling=False,
            sort_by_event_date=False,
        )
        reference_valid_df = filter_and_sample_interactions(
            valid_df,
            user_id_to_idx=self.user_id_to_idx,
            item_id_to_idx=self.item_id_to_idx,
            config=self.config,
            apply_sampling=False,
            sort_by_event_date=False,
        )
        return prepared_train_df, prepared_valid_df, reference_train_df, reference_valid_df

    def _prepare_side_feature_tables(
        self,
        *,
        users_df: pd.DataFrame | None,
        items_df: pd.DataFrame | None,
        user_feature_config: FeatureConfig | None,
        item_feature_config: FeatureConfig | None,
    ) -> None:
        if users_df is None and items_df is None:
            self._user_feature_tables = None
            self._item_feature_tables = None
            self._user_feature_metadata = FeatureMetadata.empty()
            self._item_feature_metadata = FeatureMetadata.empty()
            return

        if users_df is None or items_df is None:
            raise ValueError("`users_df` and `items_df` must be provided together when using side features.")

        if user_feature_config is None or item_feature_config is None:
            raise ValueError(
                "`user_feature_config` and `item_feature_config` must be provided together with `users_df` and `items_df`."
            )

        self._user_feature_tables = build_feature_tables(
            df=users_df,
            entity_ids=self.idx_to_user_id,
            config=user_feature_config,
            id_column=self.user_col,
        )
        self._item_feature_tables = build_feature_tables(
            df=items_df,
            entity_ids=self.idx_to_item_id,
            config=item_feature_config,
            id_column=self.item_col,
        )
        self._user_feature_metadata = self._user_feature_tables.metadata
        self._item_feature_metadata = self._item_feature_tables.metadata

    def _refresh_evaluation_reference_data(self) -> None:
        seen, popularity = build_evaluation_reference_data(self.train_df, None)
        self._seen_items_by_user = seen
        self._train_positive_item_ids_by_popularity = popularity

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
