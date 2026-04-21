"""Top-level public API for the twotower package."""

from .src.backend.config import TwoTowerConfig
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


import os

__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

from twotower.api import *  # noqa: F403, E402
from twotower.api import __version__  # noqa: E402


del os