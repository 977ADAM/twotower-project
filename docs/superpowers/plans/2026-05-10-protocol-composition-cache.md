# Protocol Composition & Evaluator Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate field duplication in `_Predictable`/`_Evaluable` by composing shared protocols, and route `TwoTower.recall_at_k` through the predictor's embedding cache.

**Architecture:** Three surgical edits across two retrieval modules and the core delegator. `predict.py` and `evaluate.py` each gain `_HasConfig, _HasIDMappings` as base classes and drop the fields those protocols already declare. `TwoTowerEvaluator.recall_at_k` gains an optional `item_embeddings` parameter; `TwoTower.recall_at_k` passes cached embeddings from `_predictor` so `encode_candidates` is not called twice per evaluation.

**Tech Stack:** Python, `typing.Protocol`, `torch.Tensor`, `unittest.mock`

---

## File Map

| File | Change |
|------|--------|
| `twotower/_src/retrieval/predict.py` | Compose `_HasConfig, _HasIDMappings`; remove 5 duplicated field declarations; swap import |
| `twotower/_src/retrieval/evaluate.py` | Same composition + field removal; add `item_embeddings` param to `recall_at_k` |
| `twotower/_src/core.py` | Update `recall_at_k` delegator to pass cached embeddings from `_predictor` |
| `tests/test_evaluate.py` | Add one new test: cache path skips `encode_candidates` |

---

## Task 1: Compose protocols in predict.py

**Files:**
- Modify: `twotower/_src/retrieval/predict.py:1-26`

- [ ] **Step 1: Confirm baseline passes**

```bash
.venv/bin/pytest tests/ -q
```

Expected: all green.

- [ ] **Step 2: Update imports in predict.py**

Replace the existing import block at the top of `twotower/_src/retrieval/predict.py`:

```python
# BEFORE (lines 1-8)
from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd
import torch

from twotower._src.config import _Config
```

```python
# AFTER
from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd
import torch

from twotower._src.protocols import _HasConfig, _HasIDMappings
```

- [ ] **Step 3: Replace `_Predictable` class definition**

Replace the current class (lines 11–26 of `twotower/_src/retrieval/predict.py`):

```python
# BEFORE
class _Predictable(Protocol):
    """Minimal model contract required by the prediction module."""

    config: _Config
    device: torch.device
    query_id_to_idx: dict[int, int]
    candidate_id_to_idx: dict[int, int]
    idx_to_query_id: list[int]
    idx_to_candidate_id: list[int]
    query_col: str
    candidate_col: str

    def eval(self) -> object: ...
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...
    def get_seen_candidates_by_query(self) -> dict[int, set[int]]: ...
```

```python
# AFTER
class _Predictable(_HasConfig, _HasIDMappings, Protocol):
    """Minimal model contract required by the prediction module."""

    device: torch.device
    query_col: str
    candidate_col: str

    def eval(self) -> object: ...
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...
    def get_seen_candidates_by_query(self) -> dict[int, set[int]]: ...
```

- [ ] **Step 4: Run tests to confirm still green**

```bash
.venv/bin/pytest tests/ -q
```

Expected: all green (no behaviour change — purely structural).

- [ ] **Step 5: Commit**

```bash
git add twotower/_src/retrieval/predict.py
git commit -m "refactor: compose _HasConfig, _HasIDMappings in _Predictable"
```

---

## Task 2: Compose protocols in evaluate.py + add item_embeddings parameter

**Files:**
- Modify: `twotower/_src/retrieval/evaluate.py:1-57, 151-203`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

Add this test at the bottom of `tests/test_evaluate.py`:

```python
def test_recall_at_k_uses_provided_item_embeddings_without_calling_encode_candidates(evaluator_setup):
    from unittest.mock import patch
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    # items in idx_to_candidate_id order: [10, 20, 30] → indices [0, 1, 2]
    item_embeddings = torch.stack([
        model._item_embeddings_by_idx[0],
        model._item_embeddings_by_idx[1],
        model._item_embeddings_by_idx[2],
    ])
    with patch.object(model, "encode_candidates", wraps=model.encode_candidates) as mock_encode:
        recall = evaluator.recall_at_k(
            model, evaluate_inputs.prepared_test_df, top_k=1,
            item_embeddings=item_embeddings,
        )
    assert recall == 1.0
    mock_encode.assert_not_called()
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
.venv/bin/pytest tests/test_evaluate.py::test_recall_at_k_uses_provided_item_embeddings_without_calling_encode_candidates -v
```

Expected: FAIL — `TypeError: recall_at_k() got an unexpected keyword argument 'item_embeddings'`

- [ ] **Step 3: Update imports in evaluate.py**

Replace the import block at the top of `twotower/_src/retrieval/evaluate.py`:

```python
# BEFORE (lines 1-12)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from twotower._src.config import _Config
from twotower._src.metrics import mean_recall, user_recall
```

```python
# AFTER
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from twotower._src.metrics import mean_recall, user_recall
from twotower._src.protocols import _HasConfig, _HasIDMappings
```

- [ ] **Step 4: Replace `_Evaluable` class definition**

Replace the current class (lines 27–56 of `twotower/_src/retrieval/evaluate.py`):

```python
# BEFORE
class _Evaluable(Protocol):
    """Minimal model contract required by the evaluation module."""

    config: _Config
    device: torch.device
    query_id_to_idx: dict[int, int]
    candidate_id_to_idx: dict[int, int]
    idx_to_candidate_id: list[int]

    def eval(self) -> object: ...
    def ensure_fitted(self) -> None: ...

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs: ...

    def make_loader(
        self,
        *,
        positive_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader[Any]: ...

    def resolve_eval_top_ks(self, top_k: int | None) -> list[int]: ...

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor: ...
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...

    def get_seen_candidates_by_query(self) -> dict[int, set[int]]: ...
    def get_train_positive_item_ranking(self) -> list[int]: ...
```

```python
# AFTER
class _Evaluable(_HasConfig, _HasIDMappings, Protocol):
    """Minimal model contract required by the evaluation module."""

    device: torch.device

    def eval(self) -> object: ...
    def ensure_fitted(self) -> None: ...

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs: ...

    def make_loader(
        self,
        *,
        positive_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        shuffle: bool,
    ) -> DataLoader[Any]: ...

    def resolve_eval_top_ks(self, top_k: int | None) -> list[int]: ...

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor: ...
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...

    def get_seen_candidates_by_query(self) -> dict[int, set[int]]: ...
    def get_train_positive_item_ranking(self) -> list[int]: ...
```

- [ ] **Step 5: Update `recall_at_k` signature and body**

Replace the `recall_at_k` method (lines 151–203 of `twotower/_src/retrieval/evaluate.py`):

```python
# BEFORE
def recall_at_k(
    self,
    model: _Evaluable,
    evaluation_df: pd.DataFrame,
    top_k: int,
    exclude_seen: bool = True,
) -> float:
    candidate_user_ids = self.get_eval_user_ids(model, evaluation_df)
    if not candidate_user_ids:
        return 0.0

    positive_df = evaluation_df[evaluation_df["label"] == 1.0]
    seen_candidates_by_query = model.get_seen_candidates_by_query() if exclude_seen else {}

    item_ids = list(model.idx_to_candidate_id)
    item_indices = torch.tensor(
        [model.candidate_id_to_idx[cid] for cid in item_ids],
        dtype=torch.long,
        device=model.device,
    )
    with torch.no_grad():
        item_embeddings = model.encode_candidates(item_indices)

    recalls = []
    for query_id in candidate_user_ids:
        actual_items = set(
            positive_df.loc[positive_df["query_id"] == query_id, "candidate_id"].astype(int)
        )
        if not actual_items or query_id not in model.query_id_to_idx:
            continue

        excluded = seen_candidates_by_query.get(query_id, set())
        candidate_positions = [
            pos for pos, cid in enumerate(item_ids) if cid not in excluded
        ]
        if not candidate_positions:
            recalls.append(0.0)
            continue

        user_idx = torch.tensor(
            [model.query_id_to_idx[query_id]], dtype=torch.long, device=model.device,
        )
        with torch.no_grad():
            user_embedding = model.encode_queries(user_idx).squeeze(0)

        candidate_embeddings = item_embeddings[candidate_positions]
        scores = torch.matmul(candidate_embeddings, user_embedding)
        k = min(top_k, scores.size(0))
        _, top_positions = torch.topk(scores, k=k)
        predicted_items = {item_ids[candidate_positions[p]] for p in top_positions.cpu().tolist()}
        recalls.append(user_recall(actual_items, predicted_items))

    return mean_recall(recalls)
```

```python
# AFTER
def recall_at_k(
    self,
    model: _Evaluable,
    evaluation_df: pd.DataFrame,
    top_k: int,
    exclude_seen: bool = True,
    item_embeddings: torch.Tensor | None = None,
) -> float:
    candidate_user_ids = self.get_eval_user_ids(model, evaluation_df)
    if not candidate_user_ids:
        return 0.0

    positive_df = evaluation_df[evaluation_df["label"] == 1.0]
    seen_candidates_by_query = model.get_seen_candidates_by_query() if exclude_seen else {}

    item_ids = list(model.idx_to_candidate_id)
    if item_embeddings is None:
        item_indices = torch.tensor(
            [model.candidate_id_to_idx[cid] for cid in item_ids],
            dtype=torch.long,
            device=model.device,
        )
        with torch.no_grad():
            item_embeddings = model.encode_candidates(item_indices)

    recalls = []
    for query_id in candidate_user_ids:
        actual_items = set(
            positive_df.loc[positive_df["query_id"] == query_id, "candidate_id"].astype(int)
        )
        if not actual_items or query_id not in model.query_id_to_idx:
            continue

        excluded = seen_candidates_by_query.get(query_id, set())
        candidate_positions = [
            pos for pos, cid in enumerate(item_ids) if cid not in excluded
        ]
        if not candidate_positions:
            recalls.append(0.0)
            continue

        user_idx = torch.tensor(
            [model.query_id_to_idx[query_id]], dtype=torch.long, device=model.device,
        )
        with torch.no_grad():
            user_embedding = model.encode_queries(user_idx).squeeze(0)

        candidate_embeddings = item_embeddings[candidate_positions]
        scores = torch.matmul(candidate_embeddings, user_embedding)
        k = min(top_k, scores.size(0))
        _, top_positions = torch.topk(scores, k=k)
        predicted_items = {item_ids[candidate_positions[p]] for p in top_positions.cpu().tolist()}
        recalls.append(user_recall(actual_items, predicted_items))

    return mean_recall(recalls)
```

- [ ] **Step 6: Run full test suite**

```bash
.venv/bin/pytest tests/ -q
```

Expected: all green, including the new cache test.

- [ ] **Step 7: Commit**

```bash
git add twotower/_src/retrieval/evaluate.py tests/test_evaluate.py
git commit -m "refactor: compose _HasConfig/_HasIDMappings in _Evaluable; add item_embeddings cache param to recall_at_k"
```

---

## Task 3: Update TwoTower.recall_at_k to pass cached embeddings

**Files:**
- Modify: `twotower/_src/core.py:376-377`

- [ ] **Step 1: Replace the delegator in core.py**

Find the `recall_at_k` method (currently at line 376) and replace it:

```python
# BEFORE
def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float:
    return self._evaluator.recall_at_k(self, evaluation_df, top_k, exclude_seen=exclude_seen)
```

```python
# AFTER
def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float:
    item_embeddings, _ = self._predictor.get_candidate_item_embeddings(
        self, list(self.idx_to_candidate_id)
    )
    return self._evaluator.recall_at_k(
        self, evaluation_df, top_k, exclude_seen=exclude_seen,
        item_embeddings=item_embeddings,
    )
```

- [ ] **Step 2: Run full test suite**

```bash
.venv/bin/pytest tests/ -q
```

Expected: all green.

- [ ] **Step 3: Commit**

```bash
git add twotower/_src/core.py
git commit -m "perf: route TwoTower.recall_at_k through predictor embedding cache"
```
