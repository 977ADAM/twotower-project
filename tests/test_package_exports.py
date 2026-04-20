from __future__ import annotations

import unittest

import twotower


class PackageExportTests(unittest.TestCase):
    def test_top_level_exports_are_explicit_and_import_star_safe(self):
        self.assertEqual(twotower.__all__, ["TwoTower", "TwoTowerConfig"])

        namespace: dict[str, object] = {}
        exec("from twotower import *", {}, namespace)

        self.assertIs(namespace["TwoTower"], twotower.TwoTower)
        self.assertIs(namespace["TwoTowerConfig"], twotower.TwoTowerConfig)
        self.assertNotIn("TwoTowerBase", namespace)


if __name__ == "__main__":
    unittest.main()
