# Changelog

## [Unreleased]

### Changed
- `features.py`: removed all hardcoded column constants (`USER_REQUIRED_COLUMNS`, `ITEM_REQUIRED_COLUMNS`, `USER_SCALAR_FEATURE_NAMES`, etc.) and dataset-specific `_bucketize_age` transform; `build_user_feature_tables` / `build_item_feature_tables` replaced by generic `build_feature_tables(df, entity_ids, config, id_column)`
- `TwoTower.fit()` now accepts `user_feature_config: FeatureConfig | None` and `item_feature_config: FeatureConfig | None` — side-feature schema is no longer hardcoded inside the library
- `core.py` now orchestrates only; data preparation logic moved to `preprocessing.py`
- `TwoTower` implements service protocols directly — `_TwoTowerBackend` adapter removed
- `pyproject.toml`: package renamed `twotower-project` → `twotower`; app dependencies (`fastapi`, `gradio`, `uvicorn`, `pydantic`) moved to `[project.optional-dependencies] app`
- Added `ruff`, `mypy`, `pytest` to `[project.optional-dependencies] dev`

### Added
- `EarlyStopping` dataclass — passed to `fit()` as a parameter; supports `metric="valid_loss"` (minimize) or `metric="recall_at_<k>"` (maximize); `patience` and `min_delta` are configurable; pass `early_stopping=None` to disable
- `TwoTowerConfig.eval_during_training` — when `True` (now the default), recall@k is computed on the validation set after each training epoch and included in the history
- `FeatureConfig` and `MultiFeatureSpec` dataclasses — exported from `twotower` public API
- `src/data.py`: `bucketize_age` utility for application-side age bucketing
- `twotower/preprocessing.py` — pure functions for data preparation
- `twotower.__version__`
- GitHub Actions CI (`.github/workflows/ci.yml`)
- `ruff` and `mypy` config in `pyproject.toml`
