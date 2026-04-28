from __future__ import annotations

import pytest

from twotower._src.config import TwoTowerConfig


def test_default_config_is_valid():
    config = TwoTowerConfig()
    assert config.epochs == 25
    assert config.batch_size == 2048
    assert config.device == "cpu"


def test_config_is_immutable():
    config = TwoTowerConfig()
    with pytest.raises(Exception):
        config.epochs = 10  # type: ignore[misc]


@pytest.mark.parametrize("field,value", [
    ("epochs", 0),
    ("epochs", -1),
    ("batch_size", 0),
    ("hidden_dim", 0),
    ("user_embedding_dim", -1),
    ("item_embedding_dim", -1),
    ("side_feature_embedding_dim", -1),
    ("retrieval_temperature", 0.0),
    ("retrieval_temperature", -0.1),
    ("learning_rate", -0.01),
    ("max_samples", 0),
    ("max_samples", -100),
    ("max_eval_users", 0),
    ("top_k", 0),
])
def test_config_rejects_invalid_positive_fields(field, value):
    with pytest.raises(ValueError, match=f"`{field}`"):
        TwoTowerConfig(**{field: value})


def test_config_rejects_empty_eval_top_ks():
    with pytest.raises(ValueError, match="`eval_top_ks`"):
        TwoTowerConfig(eval_top_ks=())


def test_config_rejects_nonpositive_eval_top_ks():
    with pytest.raises(ValueError, match="`eval_top_ks`"):
        TwoTowerConfig(eval_top_ks=(10, 0, 50))


def test_config_rejects_invalid_device():
    with pytest.raises(ValueError, match="`device`"):
        TwoTowerConfig(device="gpu")


def test_config_allows_none_device():
    config = TwoTowerConfig(device=None)
    assert config.device is None


def test_config_allows_zero_learning_rate():
    config = TwoTowerConfig(learning_rate=0.0)
    assert config.learning_rate == 0.0


def test_config_allows_none_max_samples():
    config = TwoTowerConfig(max_samples=None)
    assert config.max_samples is None
