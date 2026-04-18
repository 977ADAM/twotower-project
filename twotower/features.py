from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch

UNKNOWN_TOKEN = "__unk__"

USER_REQUIRED_COLUMNS = {
    "user_id",
    "age",
    "gender",
    "city_tier",
    "device_os",
    "platform",
    "income_band",
    "activity_segment",
    "interest_1",
    "interest_2",
    "interest_3",
    "is_premium",
}
ITEM_REQUIRED_COLUMNS = {
    "banner_id",
    "brand",
    "category",
    "subcategory",
    "banner_format",
    "campaign_goal",
    "target_gender",
    "target_age_min",
    "target_age_max",
}

USER_SCALAR_FEATURE_NAMES = (
    "age_bucket",
    "gender",
    "city_tier",
    "device_os",
    "platform",
    "income_band",
    "activity_segment",
    "is_premium",
)
ITEM_SCALAR_FEATURE_NAMES = (
    "brand",
    "category",
    "subcategory",
    "banner_format",
    "campaign_goal",
    "target_gender",
    "target_age_bucket",
)
USER_MULTI_FEATURE_SOURCES = {
    "interest_ids": ("interest_1", "interest_2", "interest_3"),
}


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


def build_user_feature_tables(users_df: pd.DataFrame, user_ids: list[int]) -> FeatureTables:
    _validate_columns(users_df, USER_REQUIRED_COLUMNS, entity_name="users")
    user_rows = _lookup_rows(users_df, "user_id", user_ids)

    prepared_users = pd.DataFrame(index=user_rows.index)
    prepared_users["age_bucket"] = _bucketize_age(user_rows["age"])
    prepared_users["gender"] = _normalize_categorical_series(user_rows["gender"])
    prepared_users["city_tier"] = _normalize_categorical_series(user_rows["city_tier"])
    prepared_users["device_os"] = _normalize_categorical_series(user_rows["device_os"])
    prepared_users["platform"] = _normalize_categorical_series(user_rows["platform"])
    prepared_users["income_band"] = _normalize_categorical_series(user_rows["income_band"])
    prepared_users["activity_segment"] = _normalize_categorical_series(user_rows["activity_segment"])
    prepared_users["is_premium"] = _normalize_categorical_series(user_rows["is_premium"])

    scalar_features: dict[str, torch.Tensor] = {}
    vocab_sizes: dict[str, int] = {}
    for feature_name in USER_SCALAR_FEATURE_NAMES:
        encoded_feature, vocab_size = _encode_scalar_feature(prepared_users[feature_name])
        scalar_features[feature_name] = encoded_feature
        vocab_sizes[feature_name] = vocab_size

    interest_frame = user_rows.loc[:, USER_MULTI_FEATURE_SOURCES["interest_ids"]].copy()
    encoded_interests, interest_vocab_size = _encode_multi_feature(interest_frame)

    metadata = FeatureMetadata(
        scalar_feature_names=USER_SCALAR_FEATURE_NAMES,
        multi_feature_names=("interest_ids",),
        multi_feature_widths={"interest_ids": len(USER_MULTI_FEATURE_SOURCES["interest_ids"])},
        vocab_sizes={
            **vocab_sizes,
            "interest_ids": interest_vocab_size,
        },
    )
    return FeatureTables(
        scalar_features=scalar_features,
        multi_features={"interest_ids": encoded_interests},
        metadata=metadata,
    )


def build_item_feature_tables(items_df: pd.DataFrame, item_ids: list[int]) -> FeatureTables:
    _validate_columns(items_df, ITEM_REQUIRED_COLUMNS, entity_name="items")
    item_rows = _lookup_rows(items_df, "banner_id", item_ids)

    prepared_items = pd.DataFrame(index=item_rows.index)
    prepared_items["brand"] = _normalize_categorical_series(item_rows["brand"])
    prepared_items["category"] = _normalize_categorical_series(item_rows["category"])
    prepared_items["subcategory"] = _normalize_categorical_series(item_rows["subcategory"])
    prepared_items["banner_format"] = _normalize_categorical_series(item_rows["banner_format"])
    prepared_items["campaign_goal"] = _normalize_categorical_series(item_rows["campaign_goal"])
    prepared_items["target_gender"] = _normalize_categorical_series(item_rows["target_gender"])
    prepared_items["target_age_bucket"] = _bucketize_age(
        (
            pd.to_numeric(item_rows["target_age_min"], errors="coerce")
            + pd.to_numeric(item_rows["target_age_max"], errors="coerce")
        )
        / 2.0
    )

    scalar_features: dict[str, torch.Tensor] = {}
    vocab_sizes: dict[str, int] = {}
    for feature_name in ITEM_SCALAR_FEATURE_NAMES:
        encoded_feature, vocab_size = _encode_scalar_feature(prepared_items[feature_name])
        scalar_features[feature_name] = encoded_feature
        vocab_sizes[feature_name] = vocab_size

    metadata = FeatureMetadata(
        scalar_feature_names=ITEM_SCALAR_FEATURE_NAMES,
        multi_feature_names=(),
        multi_feature_widths={},
        vocab_sizes=vocab_sizes,
    )
    return FeatureTables(
        scalar_features=scalar_features,
        multi_features={},
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


def _bucketize_age(values: pd.Series) -> pd.Series:
    numeric_values = pd.to_numeric(values, errors="coerce")
    bucketed = pd.cut(
        numeric_values,
        bins=[-1, 24, 34, 44, 54, float("inf")],
        labels=["18_24", "25_34", "35_44", "45_54", "55_plus"],
    )
    return _normalize_categorical_series(pd.Series(bucketed, index=values.index))


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
