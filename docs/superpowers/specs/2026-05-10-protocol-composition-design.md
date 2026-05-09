# Protocol Composition & Evaluator Cache Design

**Goal:** Eliminate field duplication in `_Predictable`/`_Evaluable` by composing shared protocols, and route `TwoTower.recall_at_k` through the predictor's embedding cache.

**Architecture:** Two targeted changes to `retrieval/predict.py`, `retrieval/evaluate.py`, and `core.py`. No new files, no new protocols.

**Tech Stack:** Python, typing.Protocol, torch.Tensor

---

## Change 1: Protocol Composition

### Problem

`_Predictable` (in `predict.py`) and `_Evaluable` (in `evaluate.py`) both redeclare fields already captured by the shared protocols in `protocols.py`:

- `config: _Config` — already in `_HasConfig`
- `query_id_to_idx`, `candidate_id_to_idx`, `idx_to_query_id`, `idx_to_candidate_id` — already in `_HasIDMappings`

Only `device: torch.device`, `encode_queries`, and `encode_candidates` are genuinely local to these protocols.

### Fix

Both protocols compose `_HasConfig` and `_HasIDMappings` and declare only what's unique:

```python
# predict.py
class _Predictable(_HasConfig, _HasIDMappings, Protocol):
    device: torch.device
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...
```

```python
# evaluate.py
class _Evaluable(_HasConfig, _HasIDMappings, Protocol):
    device: torch.device
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...
    def get_seen_candidates_by_query(self) -> dict[int, set[int]]: ...
    def get_train_positive_item_ranking(self) -> list[int]: ...
```

**Imports added:** `from twotower._src.protocols import _HasConfig, _HasIDMappings` in both files.

**Imports removed:** `from twotower._src.config import _Config` in predict.py and evaluate.py (if used only for the protocol — verify before removing).

---

## Change 2: Embedding Cache in `recall_at_k`

### Problem

`TwoTowerEvaluator.recall_at_k` calls `model.encode_candidates(item_indices)` directly, re-computing all item embeddings on every call. `TwoTowerPredictor` already maintains a cache (`_cached_all_item_embeddings`) that `TwoTower.recall_at_k` ignores.

### Fix

**`TwoTowerEvaluator.recall_at_k`** gains an optional `item_embeddings` parameter:

```python
def recall_at_k(
    self,
    model: _Evaluable,
    evaluation_df: pd.DataFrame,
    top_k: int,
    exclude_seen: bool = True,
    item_embeddings: torch.Tensor | None = None,
) -> float:
    ...
    item_ids = list(model.idx_to_candidate_id)
    if item_embeddings is None:
        item_indices = torch.tensor(
            [model.candidate_id_to_idx[cid] for cid in item_ids],
            dtype=torch.long,
            device=model.device,
        )
        with torch.no_grad():
            item_embeddings = model.encode_candidates(item_indices)
    # rest of method unchanged
```

When `item_embeddings` is `None` (default), behaviour is identical to before — no regression for callers that don't pass it.

**`TwoTower.recall_at_k`** (delegator in `core.py`) passes cached embeddings:

```python
def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float:
    item_embeddings, _ = self._predictor.get_candidate_item_embeddings(
        self, list(self.idx_to_candidate_id)
    )
    return self._evaluator.recall_at_k(
        self, evaluation_df, top_k, exclude_seen=exclude_seen,
        item_embeddings=item_embeddings,
    )
```

**Cache lifecycle** (unchanged from existing design):
- First `recall_at_k` call after training: predictor builds and caches embeddings.
- Subsequent calls: predictor returns cached embeddings, `encode_candidates` not called.
- After `fit()`: `invalidate_item_embedding_cache()` is called automatically — next call rebuilds.

---

## Files Changed

| File | Change |
|------|--------|
| `twotower/_src/retrieval/predict.py` | Compose `_HasConfig, _HasIDMappings`; remove duplicated field declarations |
| `twotower/_src/retrieval/evaluate.py` | Compose `_HasConfig, _HasIDMappings`; remove duplicated fields; add `item_embeddings` param |
| `twotower/_src/core.py` | Update `recall_at_k` delegator to pass cached embeddings |

---

## Tests

All existing tests must continue to pass unchanged.

**New test in `tests/test_evaluate.py`:**

`test_recall_at_k_uses_provided_item_embeddings_without_calling_encode_candidates` — verify that when `item_embeddings` is passed, `model.encode_candidates` is never called. Uses a stub model with a spy on `encode_candidates`.
