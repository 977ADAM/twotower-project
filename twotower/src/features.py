from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

UNKNOWN_TOKEN = "__unk__"


@dataclass(slots=True, frozen=True)
class MultiFeatureSpec:
    """Describes a single multi-valued feature and its source columns."""

    name: str
    columns: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class FeatureConfig:
    """Declares which columns to encode as side features."""

    scalar_features: tuple[str, ...] = ()
    multi_features: tuple[MultiFeatureSpec, ...] = ()


@dataclass(slots=True)
class FeatureMetadata:
    scalar_feature_names: tuple[str, ...]
    multi_feature_names: tuple[str, ...]
    multi_feature_widths: dict[str, int]
    vocab_sizes: dict[str, int]

    @classmethod
    def empty(cls) -> "FeatureMetadata":
        return cls(
            scalar_feature_names=(),
            multi_feature_names=(),
            multi_feature_widths={},
            vocab_sizes={},
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "scalar_feature_names": list(self.scalar_feature_names),
            "multi_feature_names": list(self.multi_feature_names),
            "multi_feature_widths": dict(self.multi_feature_widths),
            "vocab_sizes": dict(self.vocab_sizes),
        }

    @classmethod
    def from_dict(cls, data: object) -> "FeatureMetadata":
        if not isinstance(data, dict):
            return cls.empty()

        return cls(
            scalar_feature_names=tuple(str(name) for name in data.get("scalar_feature_names", [])),
            multi_feature_names=tuple(str(name) for name in data.get("multi_feature_names", [])),
            multi_feature_widths={
                str(name): int(width)
                for name, width in dict(data.get("multi_feature_widths", {})).items()
            },
            vocab_sizes={
                str(name): int(size)
                for name, size in dict(data.get("vocab_sizes", {})).items()
            },
        )


@dataclass(slots=True)
class FeatureTables:
    scalar_features: dict[str, torch.Tensor]
    multi_features: dict[str, torch.Tensor]
    metadata: FeatureMetadata


def build_feature_tables(
    df: pd.DataFrame,
    entity_ids: list[int],
    config: FeatureConfig,
    id_column: str,
) -> FeatureTables:
    """Encode side features for the given entity ids according to `config`.

    The caller is responsible for preparing all columns referenced in `config`
    before calling this function. Column values are treated as categorical
    strings; no numeric transforms are applied inside the library.
    """
    required_columns: set[str] = {id_column} | set(config.scalar_features)
    for spec in config.multi_features:
        required_columns.update(spec.columns)
    _validate_columns(df, required_columns, entity_name=id_column)

    rows = _lookup_rows(df, id_column, entity_ids)

    scalar_features: dict[str, torch.Tensor] = {}
    vocab_sizes: dict[str, int] = {}
    for feature_name in config.scalar_features:
        encoded, vocab_size = _encode_scalar_feature(rows[feature_name])
        scalar_features[feature_name] = encoded
        vocab_sizes[feature_name] = vocab_size

    multi_features: dict[str, torch.Tensor] = {}
    multi_feature_widths: dict[str, int] = {}
    for spec in config.multi_features:
        multi_frame = rows.loc[:, list(spec.columns)].copy()
        encoded, vocab_size = _encode_multi_feature(multi_frame)
        multi_features[spec.name] = encoded
        vocab_sizes[spec.name] = vocab_size
        multi_feature_widths[spec.name] = len(spec.columns)

    metadata = FeatureMetadata(
        scalar_feature_names=config.scalar_features,
        multi_feature_names=tuple(spec.name for spec in config.multi_features),
        multi_feature_widths=multi_feature_widths,
        vocab_sizes=vocab_sizes,
    )
    return FeatureTables(
        scalar_features=scalar_features,
        multi_features=multi_features,
        metadata=metadata,
    )


def _validate_columns(dataframe: pd.DataFrame, required_columns: set[str], entity_name: str) -> None:
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        raise ValueError(
            f"{entity_name} dataframe is missing required columns: {sorted(missing_columns)}"
        )


def _lookup_rows(dataframe: pd.DataFrame, id_column: str, entity_ids: list[int]) -> pd.DataFrame:
    deduplicated = dataframe.drop_duplicates(subset=[id_column], keep="last")
    indexed = deduplicated.set_index(id_column)
    return indexed.reindex(entity_ids)


def _normalize_categorical_series(values: pd.Series) -> pd.Series:
    normalized = values.astype("string")
    normalized = normalized.fillna(UNKNOWN_TOKEN)
    normalized = normalized.str.strip()
    normalized = normalized.mask(normalized == "", UNKNOWN_TOKEN)
    return normalized.astype(str)


def _encode_scalar_feature(values: pd.Series) -> tuple[torch.Tensor, int]:
    normalized_values = _normalize_categorical_series(values)
    vocabulary = _build_vocabulary(normalized_values.to_numpy().tolist())
    encoded_values = torch.tensor(
        [vocabulary.get(value, 0) for value in normalized_values.to_numpy().tolist()],
        dtype=torch.long,
    )
    return encoded_values, len(vocabulary)


def _encode_multi_feature(values: pd.DataFrame) -> tuple[torch.Tensor, int]:
    normalized_frame = values.apply(_normalize_categorical_series)
    vocabulary = _build_vocabulary(normalized_frame.to_numpy().reshape(-1).tolist())
    encoded_rows = [
        [vocabulary.get(value, 0) for value in row]
        for row in normalized_frame.to_numpy().tolist()
    ]
    return torch.tensor(encoded_rows, dtype=torch.long), len(vocabulary)


def _build_vocabulary(values: list[str]) -> dict[str, int]:
    vocabulary = {UNKNOWN_TOKEN: 0}
    unique_values = sorted({value for value in values if value != UNKNOWN_TOKEN})
    for value in unique_values:
        vocabulary[value] = len(vocabulary)
    return vocabulary
