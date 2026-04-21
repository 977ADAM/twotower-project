from __future__ import annotations

import twotower


def test_top_level_exports_are_explicit_and_import_star_safe():
    assert twotower.__all__ == [
        "TwoTower", "TwoTowerConfig", "FeatureConfig", "MultiFeatureSpec",
        "EarlyStopping", "NegativeSampling", "__version__",
    ]

    namespace: dict[str, object] = {}
    exec("from twotower import *", {}, namespace)

    assert namespace["TwoTower"] is twotower.TwoTower
    assert namespace["TwoTowerConfig"] is twotower.TwoTowerConfig
    assert namespace["FeatureConfig"] is twotower.FeatureConfig
    assert namespace["MultiFeatureSpec"] is twotower.MultiFeatureSpec
    assert namespace["EarlyStopping"] is twotower.EarlyStopping
    assert namespace["NegativeSampling"] is twotower.NegativeSampling
    assert "TwoTowerBase" not in namespace
