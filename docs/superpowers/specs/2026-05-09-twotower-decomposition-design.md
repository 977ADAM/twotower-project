# TwoTower Decomposition — Design Spec

**Date:** 2026-05-09  
**Goal:** Reduce `TwoTower` from a God class (~660 lines) to a thin facade (~200 lines) by redistributing responsibilities to existing service objects and introducing focused Protocol types.

---

## Motivation

Three problems in the current design:
1. **Readability** — `core.py` mixes orchestration, retrieval, evaluation, embedding cache, and protocol implementation in one 660-line class.
2. **Testability** — services can't be tested without instantiating a full `TwoTower`; protocols require too much stubbing.
3. **Extensibility** — adding a new backend (FAISS, different loss) requires understanding all of `TwoTower`.

---

## Approach: Aggressive responsibility transfer to services

No new user-facing types. Public API (`fit`, `retrieve`, `evaluate`, `save_model`, `load_model`) is unchanged.

---

## Section 1: What moves to each service

### `TwoTowerEvaluator` gains

Currently methods on `TwoTower`, moved to `TwoTowerEvaluator` as functions/methods that take a model argument:

- `recall_at_k(model, evaluation_df, top_k, exclude_seen) -> float`
- `popularity_recall_at_k(model, evaluation_df, top_k) -> float`
- `get_eval_user_ids(model, evaluation_df) -> list[int]`
- `evaluate_loader(model, loader, prefix) -> dict[str, float]`

### `TwoTowerPredictor` gains

Embedding cache state and retrieval helpers move into `TwoTowerPredictor`:

- `_cached_all_item_embeddings: Tensor | None` — lives as instance state on predictor
- `_cached_all_item_ids: list[int] | None` — lives as instance state on predictor
- `_build_candidate_item_embeddings(model) -> tuple[Tensor, list[int]]` — private method
- `get_candidate_item_embeddings(model, item_ids) -> tuple[Tensor, list[int]]` — used by evaluator and trainer
- `get_user_embedding(model, query_id) -> Tensor`
- `invalidate_cache() -> None` — `TwoTower.invalidate_item_embedding_cache()` becomes `self._predictor.invalidate_cache()`

### What stays in `TwoTower`

Constructor, 5 public API methods, `build_towers`, `recall_at_k` (delegates to evaluator — required by `_Trainable`), `validate_checkpoint`, `apply_loaded_checkpoint_state`, `ensure_fitted`, `resolve_device`, private orchestration helpers (`_prepare_fit_inputs`, `_prepare_valid_inputs`, `_prepare_side_feature_tables`, `_refresh_evaluation_reference_data`), and all state fields (mappings, feature tables, config, history, `train_df`, `valid_df`).

`get_seen_candidates_by_query` and `get_train_positive_item_ranking` are kept as methods on `TwoTower` (the evaluator calls them via the model argument) but are not part of the documented public API. They stay out of `__init__.py` and docs.

---

## Section 2: Protocol decomposition

New file: `twotower/_src/protocols.py`

```python
class _HasEmbeddings(Protocol):
    def encode_queries(self, user_input: Tensor) -> Tensor: ...
    def encode_candidates(self, item_input: Tensor) -> Tensor: ...
    def score_pairs(self, user_input: Tensor, item_input: Tensor) -> Tensor: ...
    def retrieval_logits(self, user_input: Tensor, item_input: Tensor) -> Tensor: ...

class _HasIDMappings(Protocol):
    query_id_to_idx: dict[int, int]
    candidate_id_to_idx: dict[int, int]
    idx_to_query_id: list[int]
    idx_to_candidate_id: list[int]

class _HasConfig(Protocol):
    config: _Config
```

`_Trainable` in `fit.py` inherits all three plus `build_towers`, `train`, `eval`, `parameters`, `state_dict`, `load_state_dict`, `to`, `recall_at_k`.

### Service dependencies

| Service | Protocol |
|---|---|
| `TwoTowerTrainer` | `_Trainable` |
| `TwoTowerPredictor` | `_HasEmbeddings & _HasIDMappings & _HasConfig` |
| `TwoTowerEvaluator` | `_HasEmbeddings & _HasIDMappings & _HasConfig` |

`recall_at_k` stays in `_Trainable` because `TwoTowerTrainer` calls it for recall-based early stopping. The implementation in `TwoTower` delegates to `self._evaluator.recall_at_k(self, ...)`.

---

## Section 3: `TwoTower` after refactor

```
class TwoTower(TwoTowerBase):
    ── 5 public API methods ──────────────────────────────────────
    fit(...)           orchestration only
    retrieve(...)      → self._predictor.predict(self, ...)
    evaluate(...)      → self._evaluator.evaluate(self, ...)
    save_model(...)    → self._model_saver.save_model(self, ...)
    load_model(...)    → self._model_loader.load_model(self, ...)

    ── Protocol methods (called by services) ────────────────────
    build_towers(num_users, num_items)
    recall_at_k(...)        delegates to self._evaluator
    validate_checkpoint(...)
    apply_loaded_checkpoint_state(...)
    ensure_fitted()

    ── Private fit orchestration ────────────────────────────────
    _prepare_fit_inputs(...)
    _prepare_valid_inputs(...)
    _prepare_side_feature_tables(...)
    _refresh_evaluation_reference_data(...)

    ── Utility ──────────────────────────────────────────────────
    resolve_device()        @staticmethod
```

---

## File impact summary

| File | Change |
|---|---|
| `twotower/_src/core.py` | ~660 → ~200 lines |
| `twotower/_src/retrieval/evaluate.py` | + `recall_at_k`, `popularity_recall_at_k`, `get_eval_user_ids`, `evaluate_loader` |
| `twotower/_src/retrieval/predict.py` | + embedding cache state and methods |
| `twotower/_src/training/fit.py` | `_Trainable` composed from sub-protocols |
| `twotower/_src/protocols.py` | new file: `_HasEmbeddings`, `_HasIDMappings`, `_HasConfig` |

---

## What is not changing

- Public API: `TwoTower`, `split_interactions` — untouched
- Checkpoint format — untouched
- Training logic (`TwoTowerTrainer`) — untouched
- Test behaviour — all 105 existing tests must continue to pass
