from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from rich.console import Console
from torch.utils.data import DataLoader

from twotower._src.config import _Config
from twotower._src.data.features import (
    FeatureConfig,
    FeatureMetadata,
    FeatureTables,
    MultiFeatureSpec,
    build_feature_tables,
)
from twotower._src.data.preprocessing import (
    build_evaluation_reference_data,
    build_id_mappings,
    filter_and_sample_interactions,
    normalize_and_filter_interactions,
    normalize_fit_interactions,
    prepare_evaluation_inputs,
    prepare_retrieval_pairs,
)
from twotower._src.io.load import LoadedCheckpointState, TwoTowerModelLoader
from twotower._src.io.save import TwoTowerModelSaver
from twotower._src.nn import Tower, TwoTowerBase
from twotower._src.retrieval.evaluate import EvaluateInputs, TwoTowerEvaluator
from twotower._src.retrieval.predict import TwoTowerPredictor
from twotower._src.training.fit import (
    EarlyStopping,
    FitInputs,
    FitResult,
    NegativeSampling,
    TwoTowerTrainer,
    build_pairwise_loader,
)
from twotower._src.utils.traceback_utils import filter_traceback

console = Console()


def _build_feature_config(
    scalar_features: list[str] | None,
    multi_features: dict[str, list[str]] | None,
) -> FeatureConfig:
    return FeatureConfig(
        scalar_features=tuple(scalar_features or []),
        multi_features=tuple(
            MultiFeatureSpec(name=name, columns=tuple(cols))
            for name, cols in (multi_features or {}).items()
        ),
    )


class TwoTower(TwoTowerBase):
    def __init__(
        self,
        *,
        query_embedding_dim: int = 64,
        candidate_embedding_dim: int = 64,
        side_feature_embedding_dim: int = 8,
        hidden_dim: int = 64,
        tower_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.query_embedding_dim = query_embedding_dim
        self.candidate_embedding_dim = candidate_embedding_dim
        self.side_feature_embedding_dim = side_feature_embedding_dim
        self.hidden_dim = hidden_dim
        self.tower_dims = tower_dims
        self.dropout = dropout
        self.query_col: str = "query_id"
        self.candidate_col: str = "candidate_id"
        self.device: torch.device = torch.device("cpu")
        self.query_id_to_idx: dict[int, int] = {}
        self.candidate_id_to_idx: dict[int, int] = {}
        self.idx_to_query_id: list[int] = []
        self.idx_to_candidate_id: list[int] = []
        self.train_history: list[dict[str, float]] = []
        self.train_df: pd.DataFrame | None = None
        self.valid_df: pd.DataFrame | None = None
        self._seen_candidates_by_query: dict[int, set[int]] = {}
        self._train_positive_candidate_ids_by_popularity: list[int] = []
        self._query_feature_tables: FeatureTables | None = None
        self._candidate_feature_tables: FeatureTables | None = None
        self._query_feature_metadata: FeatureMetadata = FeatureMetadata.empty()
        self._candidate_feature_metadata: FeatureMetadata = FeatureMetadata.empty()
        self._negative_sampling: NegativeSampling = NegativeSampling()
        self._evaluator = TwoTowerEvaluator()
        self._predictor = TwoTowerPredictor()
        self._model_saver = TwoTowerModelSaver()
        self._model_loader = TwoTowerModelLoader()

    @filter_traceback
    def fit(
        self,
        train_df: pd.DataFrame,
        *,
        validation_data: pd.DataFrame | None = None,
        query_col: str = "query_id",
        candidate_col: str = "candidate_id",
        queries_df: pd.DataFrame | None = None,
        candidates_df: pd.DataFrame | None = None,
        query_features: list[str] | None = None,
        query_multi_features: dict[str, list[str]] | None = None,
        candidate_features: list[str] | None = None,
        candidate_multi_features: dict[str, list[str]] | None = None,
        observed_ratio: float = 0.8,
        in_batch_loss_weight: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 2048,
        epochs: int = 25,
        retrieval_temperature: float = 0.1,
        eval_top_ks: tuple[int, ...] = (50, 100, 300),
        max_eval_users: int = 500,
        top_k: int = 100,
        eval_during_training: bool = True,
        seed: int = 42,
        device: str | None = "cpu",
        patience: int | None = 5,
        early_stopping_metric: str = "valid_loss",
        min_delta: float = 1e-4,
    ) -> FitResult:
        """Fit the model on interaction pairs."""
        self.query_col = query_col
        self.candidate_col = candidate_col
        self.device = self.resolve_device(device)
        negative_sampling = NegativeSampling(observed_ratio=observed_ratio, in_batch_loss_weight=in_batch_loss_weight)
        self.config = _Config(
            query_embedding_dim=self.query_embedding_dim,
            candidate_embedding_dim=self.candidate_embedding_dim,
            side_feature_embedding_dim=self.side_feature_embedding_dim,
            hidden_dim=self.hidden_dim,
            tower_dims=self.tower_dims,
            dropout=self.dropout,
            retrieval_temperature=retrieval_temperature,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            eval_top_ks=eval_top_ks,
            max_eval_users=max_eval_users,
            top_k=top_k,
            eval_during_training=eval_during_training,
            seed=seed,
            device=device,
        )
        if patience is not None and validation_data is None:
            raise ValueError(
                "Cannot use early stopping without validation_data. "
                "Pass validation_data or set patience=None."
            )
        early_stopping = (
            EarlyStopping(patience=patience, metric=early_stopping_metric, min_delta=min_delta)
            if patience is not None
            else None
        )

        prepared_train_df, reference_train_df = self._prepare_fit_inputs(train_df=train_df)
        prepared_valid_df, reference_valid_df = (
            self._prepare_valid_inputs(valid_df=validation_data)
            if validation_data is not None
            else (None, None)
        )
        self.train_df = reference_train_df
        self.valid_df = reference_valid_df
        self._refresh_evaluation_reference_data()
        self._prepare_side_feature_tables(
            queries_df=queries_df,
            candidates_df=candidates_df,
            query_features=query_features,
            query_multi_features=query_multi_features,
            candidate_features=candidate_features,
            candidate_multi_features=candidate_multi_features,
        )
        self._negative_sampling = negative_sampling

        self.invalidate_item_embedding_cache()

        fit_inputs = FitInputs(
            train_positive_df=prepared_train_df,
            train_interactions_df=reference_train_df,
            num_users=len(self.idx_to_query_id),
            num_items=len(self.idx_to_candidate_id),
            valid_positive_df=prepared_valid_df,
            valid_interactions_df=reference_valid_df,
        )
        trainer = TwoTowerTrainer(config=self.config, device=self.device)

        fit_result = trainer.fit(
            self, fit_inputs,
            negative_sampling=negative_sampling,
            early_stopping=early_stopping,
        )
        self.train_history = fit_result.history

        self.invalidate_item_embedding_cache()

        return fit_result

    @filter_traceback
    def retrieve(
        self,
        user_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        exclude_seen: bool = True,
        strict: bool = False,
    ) -> pd.DataFrame:
        """Return top-k item recommendations for the requested users.

        Args:
            user_ids: User IDs to generate recommendations for. If ``None``,
                up to 10 known users are used.
            item_ids: Candidate item IDs. If ``None``, all known items are used.
            top_k: Number of items to return per user. Defaults to the model's
                ``top_k`` config value.
            exclude_seen: Whether to exclude items the user interacted with
                during training.
            strict: If ``True``, raises ``ValueError`` for unknown user or item IDs.
                If ``False``, they are silently skipped.

        Returns:
            DataFrame with columns [query_col, candidate_col, "score", "rank"],
            sorted by user and rank.
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

        required_keys = {"config", "state_dict", "query_id_to_idx", "candidate_id_to_idx", "idx_to_query_id", "idx_to_candidate_id"}
        missing_keys = required_keys.difference(checkpoint)
        if missing_keys:
            raise ValueError(
                f"Invalid checkpoint format in {checkpoint_path}: missing keys {sorted(missing_keys)}."
            )

    def apply_loaded_checkpoint_state(self, state: LoadedCheckpointState) -> None:
        self.config = state.config
        self.query_embedding_dim = state.config.query_embedding_dim
        self.candidate_embedding_dim = state.config.candidate_embedding_dim
        self.side_feature_embedding_dim = state.config.side_feature_embedding_dim
        self.hidden_dim = state.config.hidden_dim
        self.tower_dims = state.config.tower_dims
        self.dropout = state.config.dropout
        self.query_col = state.query_col
        self.candidate_col = state.candidate_col
        self.device = state.device
        self.query_id_to_idx = state.query_id_to_idx
        self.candidate_id_to_idx = state.candidate_id_to_idx
        self.idx_to_query_id = state.idx_to_query_id
        self.idx_to_candidate_id = state.idx_to_candidate_id
        self.train_history = state.train_history
        self.train_df = None
        self.valid_df = None
        self._seen_candidates_by_query = state.seen_candidates_by_query
        self._train_positive_candidate_ids_by_popularity = state.train_positive_candidate_ids_by_popularity
        self._query_feature_tables = None
        self._candidate_feature_tables = None
        self._query_feature_metadata = state.query_feature_metadata
        self._candidate_feature_metadata = state.candidate_feature_metadata

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs:
        test_input_df = prepare_evaluation_inputs(X_test, self.query_col, self.candidate_col)
        prepared_test_df = normalize_and_filter_interactions(
            test_input_df,
            query_id_to_idx=self.query_id_to_idx,
            candidate_id_to_idx=self.candidate_id_to_idx,
            config=self.config,
        )
        positive_test_df = (
            prepared_test_df.copy()
            if prepared_test_df.empty
            else prepare_retrieval_pairs(
                prepared_test_df,
                query_id_to_idx=self.query_id_to_idx,
                candidate_id_to_idx=self.candidate_id_to_idx,
                config=self.config,
                split_name="test",
            )
        )
        return EvaluateInputs(
            test_input_df=test_input_df,
            prepared_test_df=prepared_test_df,
            positive_test_df=positive_test_df,
            input_row_count=len(test_input_df),
            unknown_user_row_count=int((~test_input_df["query_id"].isin(self.query_id_to_idx)).sum()),
            unknown_item_row_count=int((~test_input_df["candidate_id"].isin(self.candidate_id_to_idx)).sum()),
        )

    def make_loader(
        self,
        *,
        positive_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader[Any]:
        return build_pairwise_loader(
            positive_df=positive_df,
            interactions_df=interactions_df,
            query_id_to_idx=self.query_id_to_idx,
            candidate_id_to_idx=self.candidate_id_to_idx,
            num_items=len(self.idx_to_candidate_id),
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            observed_negative_sampling_ratio=self._negative_sampling.observed_ratio,
            seed=self.config.seed + 2,
        )

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

    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float:
        self.ensure_fitted()
        item_embeddings, _ = self._predictor.get_candidate_item_embeddings(
            self, list(self.idx_to_candidate_id)
        )
        return self._evaluator.recall_at_k(
            self, evaluation_df, top_k, exclude_seen=exclude_seen,
            item_embeddings=item_embeddings,
        )

    def build_towers(self, num_users: int, num_items: int) -> None:
        self.query_tower = Tower(
            num_users,
            self.config.query_embedding_dim,
            self.config,
            feature_tables=self._query_feature_tables,
            feature_metadata=self._query_feature_metadata,
        )
        self.candidate_tower = Tower(
            num_items,
            self.config.candidate_embedding_dim,
            self.config,
            feature_tables=self._candidate_feature_tables,
            feature_metadata=self._candidate_feature_metadata,
        )

    def ensure_fitted(self) -> None:
        if self.query_tower is None or self.candidate_tower is None:
            raise RuntimeError("Model is not fitted yet.")

    def get_seen_candidates_by_query(self) -> dict[int, set[int]]:
        if self._seen_candidates_by_query:
            return self._seen_candidates_by_query
        self._refresh_evaluation_reference_data()
        return self._seen_candidates_by_query

    def get_train_positive_item_ranking(self) -> list[int]:
        if self._train_positive_candidate_ids_by_popularity:
            return self._train_positive_candidate_ids_by_popularity
        self._refresh_evaluation_reference_data()
        return self._train_positive_candidate_ids_by_popularity

    def get_query_feature_metadata_dict(self) -> dict[str, object]:
        return self._query_feature_metadata.to_dict()

    def get_candidate_feature_metadata_dict(self) -> dict[str, object]:
        return self._candidate_feature_metadata.to_dict()

    def invalidate_item_embedding_cache(self) -> None:
        self._predictor.invalidate_cache()

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
        train_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = normalize_fit_interactions(train_df, split_name="train", query_col=self.query_col, candidate_col=self.candidate_col)

        mappings = build_id_mappings(train_df)
        self.query_id_to_idx = mappings.query_id_to_idx
        self.candidate_id_to_idx = mappings.candidate_id_to_idx
        self.idx_to_query_id = mappings.idx_to_query_id
        self.idx_to_candidate_id = mappings.idx_to_candidate_id

        assert self.config is not None
        prepared_train_df = prepare_retrieval_pairs(
            train_df,
            query_id_to_idx=self.query_id_to_idx,
            candidate_id_to_idx=self.candidate_id_to_idx,
            config=self.config,
            split_name="train",
        )
        reference_train_df = filter_and_sample_interactions(
            train_df,
            query_id_to_idx=self.query_id_to_idx,
            candidate_id_to_idx=self.candidate_id_to_idx,
            config=self.config,
        )
        return prepared_train_df, reference_train_df

    def _prepare_valid_inputs(
        self,
        *,
        valid_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert self.config is not None
        valid_df = normalize_fit_interactions(valid_df, split_name="valid", query_col=self.query_col, candidate_col=self.candidate_col)
        prepared_valid_df = prepare_retrieval_pairs(
            valid_df,
            query_id_to_idx=self.query_id_to_idx,
            candidate_id_to_idx=self.candidate_id_to_idx,
            config=self.config,
            split_name="valid",
        )
        reference_valid_df = filter_and_sample_interactions(
            valid_df,
            query_id_to_idx=self.query_id_to_idx,
            candidate_id_to_idx=self.candidate_id_to_idx,
            config=self.config,
        )
        return prepared_valid_df, reference_valid_df

    def _prepare_side_feature_tables(
        self,
        *,
        queries_df: pd.DataFrame | None,
        candidates_df: pd.DataFrame | None,
        query_features: list[str] | None,
        query_multi_features: dict[str, list[str]] | None,
        candidate_features: list[str] | None,
        candidate_multi_features: dict[str, list[str]] | None,
    ) -> None:
        if queries_df is not None:
            query_feature_config = _build_feature_config(query_features, query_multi_features)
            self._query_feature_tables = build_feature_tables(
                df=queries_df,
                entity_ids=self.idx_to_query_id,
                config=query_feature_config,
                id_column=self.query_col,
            )
            self._query_feature_metadata = self._query_feature_tables.metadata
        else:
            self._query_feature_tables = None
            self._query_feature_metadata = FeatureMetadata.empty()

        if candidates_df is not None:
            candidate_feature_config = _build_feature_config(candidate_features, candidate_multi_features)
            self._candidate_feature_tables = build_feature_tables(
                df=candidates_df,
                entity_ids=self.idx_to_candidate_id,
                config=candidate_feature_config,
                id_column=self.candidate_col,
            )
            self._candidate_feature_metadata = self._candidate_feature_tables.metadata
        else:
            self._candidate_feature_tables = None
            self._candidate_feature_metadata = FeatureMetadata.empty()

    def _refresh_evaluation_reference_data(self) -> None:
        seen, popularity = build_evaluation_reference_data(self.train_df, None)
        self._seen_candidates_by_query = seen
        self._train_positive_candidate_ids_by_popularity = popularity
