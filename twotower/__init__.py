"""Top-level public API for the twotower package."""

from .config import TwoTowerConfig
from .core import TwoTower
from .features import FeatureConfig, MultiFeatureSpec
from .fit import EarlyStopping, NegativeSampling

__version__ = "0.1.0"

__all__ = [
    "TwoTower",
    "TwoTowerConfig",
    "FeatureConfig",
    "MultiFeatureSpec",
    "EarlyStopping",
    "NegativeSampling",
    "__version__",
]
