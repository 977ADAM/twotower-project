from __future__ import annotations

import pandas as pd
import pytest

from twotower._src.data.features import (
    FeatureConfig,
    FeatureMetadata,
    MultiFeatureSpec,
    build_feature_tables,
)


@pytest.fixture
def entity_df():
    return pd.DataFrame({
        "item_id": [10, 20, 30],
        "category": ["sports", "tech", "sports"],
        "tag_1": ["a", "b", "a"],
        "tag_2": ["x", "y", "z"],
    })


def test_build_feature_tables_encodes_scalar_feature(entity_df):
    config = FeatureConfig(scalar_features=("category",))
    tables = build_feature_tables(entity_df, entity_ids=[10, 20, 30], config=config, id_column="item_id")

    assert "category" in tables.scalar_features
    encoded = tables.scalar_features["category"]
    assert encoded.shape == (3,)
    # "sports" appears at idx 0 and 2 → same encoded value
    assert encoded[0].item() == encoded[2].item()
    assert encoded[0].item() != encoded[1].item()


def test_build_feature_tables_encodes_multi_feature(entity_df):
    config = FeatureConfig(multi_features=(MultiFeatureSpec("tags", columns=("tag_1", "tag_2")),))
    tables = build_feature_tables(entity_df, entity_ids=[10, 20, 30], config=config, id_column="item_id")

    assert "tags" in tables.multi_features
    encoded = tables.multi_features["tags"]
    assert encoded.shape == (3, 2)


def test_build_feature_tables_metadata_has_correct_vocab_sizes(entity_df):
    config = FeatureConfig(
        scalar_features=("category",),
        multi_features=(MultiFeatureSpec("tags", columns=("tag_1", "tag_2")),),
    )
    tables = build_feature_tables(entity_df, entity_ids=[10, 20, 30], config=config, id_column="item_id")

    # "category" vocab: __unk__ + sports + tech = 3
    assert tables.metadata.vocab_sizes["category"] == 3
    # "tags" vocab: __unk__ + a + b + x + y + z = 6
    assert tables.metadata.vocab_sizes["tags"] == 6
    assert tables.metadata.multi_feature_widths["tags"] == 2


def test_build_feature_tables_unknown_token_for_missing_entity():
    df = pd.DataFrame({"item_id": [10], "category": ["sports"]})
    config = FeatureConfig(scalar_features=("category",))
    # entity_id=99 is not in df → should get unknown token (index 0)
    tables = build_feature_tables(df, entity_ids=[10, 99], config=config, id_column="item_id")
    assert tables.scalar_features["category"][1].item() == 0


def test_build_feature_tables_raises_for_missing_column():
    df = pd.DataFrame({"item_id": [10]})
    config = FeatureConfig(scalar_features=("category",))
    with pytest.raises(ValueError, match="category"):
        build_feature_tables(df, entity_ids=[10], config=config, id_column="item_id")


def test_feature_metadata_round_trip():
    meta = FeatureMetadata(
        scalar_feature_names=("category",),
        multi_feature_names=("tags",),
        multi_feature_widths={"tags": 2},
        vocab_sizes={"category": 3, "tags": 6},
    )
    restored = FeatureMetadata.from_dict(meta.to_dict())
    assert restored.scalar_feature_names == meta.scalar_feature_names
    assert restored.multi_feature_names == meta.multi_feature_names
    assert restored.multi_feature_widths == meta.multi_feature_widths
    assert restored.vocab_sizes == meta.vocab_sizes


def test_feature_metadata_from_dict_handles_invalid_input():
    assert FeatureMetadata.from_dict(None) == FeatureMetadata.empty()
    assert FeatureMetadata.from_dict("bad") == FeatureMetadata.empty()
