from __future__ import annotations

import twotower


def test_top_level_exports_are_explicit_and_import_star_safe():
    assert twotower.__all__ == [
        "TwoTower", "split_interactions", "__version__",
    ]

    namespace: dict[str, object] = {}
    exec("from twotower import *", {}, namespace)

    assert namespace["TwoTower"] is twotower.TwoTower
    assert namespace["split_interactions"] is twotower.split_interactions
    assert "FeatureConfig" not in namespace
    assert "MultiFeatureSpec" not in namespace
    assert "TwoTowerBase" not in namespace
    assert "_Config" not in namespace
    assert "EarlyStopping" not in namespace
    assert "NegativeSampling" not in namespace
