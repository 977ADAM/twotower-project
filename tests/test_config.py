from __future__ import annotations

import pytest

from twotower._src.config import _Config


def test_default_config_is_valid():
    config = _Config()
    assert config.epochs == 25
    assert config.batch_size == 2048
    assert config.device == "cpu"
    assert config.tower_dims == (128, 64)
    assert config.dropout == 0.0


def test_config_is_immutable():
    config = _Config()
    with pytest.raises(Exception):
        config.epochs = 10  # type: ignore[misc]


@pytest.mark.parametrize("field,value", [
    ("epochs", 0),
    ("epochs", -1),
    ("batch_size", 0),
    ("hidden_dim", 0),
    ("query_embedding_dim", -1),
    ("candidate_embedding_dim", -1),
    ("side_feature_embedding_dim", -1),
    ("retrieval_temperature", 0.0),
    ("retrieval_temperature", -0.1),
    ("learning_rate", -0.01),
    ("weight_decay", -0.01),
    ("max_eval_users", 0),
    ("top_k", 0),
    ("dropout", -0.1),
    ("dropout", 1.0),
])
def test_config_rejects_invalid_positive_fields(field, value):
    with pytest.raises(ValueError, match=f"`{field}`"):
        _Config(**{field: value})


def test_config_rejects_empty_eval_top_ks():
    with pytest.raises(ValueError, match="`eval_top_ks`"):
        _Config(eval_top_ks=())


def test_config_rejects_nonpositive_eval_top_ks():
    with pytest.raises(ValueError, match="`eval_top_ks`"):
        _Config(eval_top_ks=(10, 0, 50))


def test_config_rejects_invalid_device():
    with pytest.raises(ValueError, match="`device`"):
        _Config(device="gpu")


def test_config_allows_none_device():
    config = _Config(device=None)
    assert config.device is None


def test_config_allows_zero_learning_rate():
    config = _Config(learning_rate=0.0)
    assert config.learning_rate == 0.0


def test_config_allows_tower_dims():
    config = _Config(tower_dims=(256, 128))
    assert config.tower_dims == (256, 128)


def test_config_rejects_nonpositive_tower_dims():
    with pytest.raises(ValueError, match="`tower_dims`"):
        _Config(tower_dims=(256, 0))


def test_config_allows_zero_dropout():
    config = _Config(dropout=0.0)
    assert config.dropout == 0.0


def test_config_allows_valid_dropout():
    config = _Config(dropout=0.5)
    assert config.dropout == 0.5


