from __future__ import annotations

import twotower


def test_top_level_exports_are_explicit_and_import_star_safe():
    assert twotower.__all__ == ["TwoTower", "TwoTowerConfig", "__version__"]

    namespace: dict[str, object] = {}
    exec("from twotower import *", {}, namespace)

    assert namespace["TwoTower"] is twotower.TwoTower
    assert namespace["TwoTowerConfig"] is twotower.TwoTowerConfig
    assert "TwoTowerBase" not in namespace
