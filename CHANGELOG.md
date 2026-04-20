# Changelog

## [Unreleased]

### Changed
- `core.py` now orchestrates only; data preparation logic moved to `preprocessing.py`
- `TwoTower` implements service protocols directly — `_TwoTowerBackend` adapter removed
- `pyproject.toml`: package renamed `twotower-project` → `twotower`; app dependencies (`fastapi`, `gradio`, `uvicorn`, `pydantic`) moved to `[project.optional-dependencies] app`
- Added `ruff`, `mypy`, `pytest` to `[project.optional-dependencies] dev`

### Added
- `twotower/preprocessing.py` — pure functions for data preparation
- `twotower.__version__`
- GitHub Actions CI (`.github/workflows/ci.yml`)
- `ruff` and `mypy` config in `pyproject.toml`
