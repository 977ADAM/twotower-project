# TwoTower Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Slim `TwoTower` from a God class (~660 lines) to a thin facade by moving embedding cache and evaluation logic into `TwoTowerPredictor` and `TwoTowerEvaluator` respectively, and formalising shared protocols.

**Architecture:** Three new shared Protocol types live in `protocols.py`; `TwoTowerPredictor` gains cache state and embedding helpers; `TwoTowerEvaluator` gains recall/popularity/loader methods; `TwoTower` keeps only orchestration and thin delegators; `_Trainable` composes from sub-protocols.

**Tech Stack:** Python 3.11, PyTorch, pandas, pytest, uv (run tests with `.venv/bin/pytest` to avoid network access)

---

## File Structure

| File | Change |
|---|---|
| `twotower/_src/protocols.py` | New — shared `_HasEmbeddings`, `_HasIDMappings`, `_HasConfig` |
| `twotower/_src/retrieval/predict.py` | `TwoTowerPredictor` gains `__init__`, cache state, `get_candidate_item_embeddings(model, item_ids)`, `get_user_embedding(model, query_id)`, `invalidate_cache()`; `_Predictable` loses `get_candidate_item_embeddings`/`get_user_embedding`, gains `encode_queries`/`encode_candidates`/`device` |
| `twotower/_src/retrieval/evaluate.py` | `TwoTowerEvaluator` gains `evaluate_loader`, `get_eval_user_ids`, `recall_at_k`, `popularity_recall_at_k`; `_Evaluable` updated accordingly |
| `twotower/_src/training/fit.py` | `_Trainable` inherits from `_HasEmbeddings`, `_HasIDMappings`, `_HasConfig` |
| `twotower/_src/core.py` | Remove moved methods; `recall_at_k` becomes a 3-line delegator; `invalidate_item_embedding_cache` delegates to predictor |
| `tests/test_predict.py` | Update stub + add cache tests |
| `tests/test_evaluate.py` | Replace stub with tensor-based one + add new tests |

---

### Task 1: Create `twotower/_src/protocols.py`

**Files:**
- Create: `twotower/_src/protocols.py`
- Create: `tests/test_protocols.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_protocols.py`:

```python
from __future__ import annotations


def test_protocols_importable():
    from twotower._src.protocols import _HasConfig, _HasEmbeddings, _HasIDMappings
    assert _HasEmbeddings is not None
    assert _HasIDMappings is not None
    assert _HasConfig is not None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/test_protocols.py -v
```

Expected: `ModuleNotFoundError` — `protocols.py` does not exist yet.

- [ ] **Step 3: Create `twotower/_src/protocols.py`**

```python
from __future__ import annotations

from typing import Protocol

import torch

from twotower._src.config import _Config


class _HasEmbeddings(Protocol):
    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor: ...
    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor: ...
    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor: ...
    def retrieval_logits(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor: ...


class _HasIDMappings(Protocol):
    query_id_to_idx: dict[int, int]
    candidate_id_to_idx: dict[int, int]
    idx_to_query_id: list[int]
    idx_to_candidate_id: list[int]


class _HasConfig(Protocol):
    config: _Config
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/test_protocols.py -v
```

Expected: 1 PASSED.

- [ ] **Step 5: Commit**

```bash
git add twotower/_src/protocols.py tests/test_protocols.py
git commit -m "feat: add shared protocol types _HasEmbeddings, _HasIDMappings, _HasConfig"
```

---

### Task 2: Move embedding cache and helpers to `TwoTowerPredictor`

`_Predictable` currently requires `get_candidate_item_embeddings` and `get_user_embedding` on the model. After this task: the predictor owns that state and logic; `_Predictable` instead requires `encode_queries`, `encode_candidates`, and `device`.

**Files:**
- Modify: `twotower/_src/retrieval/predict.py`
- Modify: `tests/test_predict.py`

- [ ] **Step 1: Write new/updated tests**

Replace the entire contents of `tests/test_predict.py` with:

```python
from __future__ import annotations

import pytest
import torch

from twotower._src.config import _Config
from twotower._src.retrieval.predict import TwoTowerPredictor


class StubPredictableModel:
    def __init__(self, candidate_col: str = "candidate_id"):
        self.config = _Config(top_k=2)
        self.device = torch.device("cpu")
        self.query_col = "query_id"
        self.candidate_col = candidate_col
        self.query_id_to_idx = {1: 0, 2: 1}
        self.candidate_id_to_idx = {10: 0, 20: 1, 30: 2}
        self.idx_to_query_id = [1, 2]
        self.idx_to_candidate_id = [10, 20, 30]
        self.eval_calls = 0
        # idx → embedding (encode_queries/encode_candidates index into these)
        self._user_embeddings_by_idx = {0: torch.tensor([1.0, 0.0]), 1: torch.tensor([0.0, 1.0])}
        self._item_embeddings_by_idx = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.8, 0.2]),
            2: torch.tensor([0.0, 1.0]),
        }
        self._seen_candidates_by_query = {1: {10}, 2: set()}

    def eval(self):
        self.eval_calls += 1

    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._user_embeddings_by_idx[int(idx)] for idx in user_input])

    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._item_embeddings_by_idx[int(idx)] for idx in item_input])

    def get_seen_candidates_by_query(self) -> dict[int, set[int]]:
        return self._seen_candidates_by_query


@pytest.fixture
def predictor_setup():
    return StubPredictableModel(), TwoTowerPredictor()


def test_predict_excludes_seen_items_by_default(predictor_setup):
    model, predictor = predictor_setup
    df = predictor.predict(model, user_ids=[1], top_k=2)

    assert model.eval_calls == 1
    assert df["candidate_id"].tolist() == [20, 30]
    assert df["rank"].tolist() == [1, 2]


def test_predict_deduplicates_ids_and_skips_unknown_ids_by_default(predictor_setup):
    model, predictor = predictor_setup
    df = predictor.predict(model, user_ids=[999, 1, 1], item_ids=[20, 20, 30, 999], top_k=5)

    assert df["query_id"].unique().tolist() == [1]
    assert df["candidate_id"].tolist() == [20, 30]


def test_predict_uses_custom_candidate_col_in_output():
    model = StubPredictableModel(candidate_col="product_id")
    predictor = TwoTowerPredictor()
    df = predictor.predict(model, user_ids=[1], top_k=2)

    assert "product_id" in df.columns
    assert "candidate_id" not in df.columns


def test_predict_strict_raises_for_unknown_ids(predictor_setup):
    model, predictor = predictor_setup
    with pytest.raises(ValueError, match=r"unknown user_ids: \[999\]"):
        predictor.predict(model, user_ids=[999], strict=True)


def test_predict_top_k_item_ids_for_user_supports_recall_style_usage(predictor_setup):
    model, predictor = predictor_setup
    # embeddings now come from the predictor, not the model
    item_embeddings, item_ids = predictor.get_candidate_item_embeddings(model, model.idx_to_candidate_id)

    predicted_item_ids = predictor.predict_top_k_item_ids_for_user(
        model, query_id=1, item_embeddings=item_embeddings,
        item_ids=item_ids, top_k=2, excluded_item_ids={10},
    )

    assert predicted_item_ids == {20, 30}


def test_get_candidate_item_embeddings_populates_cache(predictor_setup):
    model, predictor = predictor_setup
    assert predictor._cached_all_item_embeddings is None
    assert predictor._cached_all_item_ids is None

    embeddings, item_ids = predictor.get_candidate_item_embeddings(model, model.idx_to_candidate_id)

    assert predictor._cached_all_item_embeddings is not None
    assert predictor._cached_all_item_ids == [10, 20, 30]
    assert embeddings.shape == (3, 2)


def test_get_candidate_item_embeddings_reuses_cache(predictor_setup):
    model, predictor = predictor_setup
    emb1, _ = predictor.get_candidate_item_embeddings(model, model.idx_to_candidate_id)
    emb2, _ = predictor.get_candidate_item_embeddings(model, model.idx_to_candidate_id)

    assert emb1 is emb2


def test_invalidate_cache_clears_state(predictor_setup):
    model, predictor = predictor_setup
    predictor.get_candidate_item_embeddings(model, model.idx_to_candidate_id)
    assert predictor._cached_all_item_embeddings is not None

    predictor.invalidate_cache()

    assert predictor._cached_all_item_embeddings is None
    assert predictor._cached_all_item_ids is None


def test_get_user_embedding_raises_for_unknown_id(predictor_setup):
    model, predictor = predictor_setup
    with pytest.raises(KeyError, match="999"):
        predictor.get_user_embedding(model, 999)


def test_get_user_embedding_returns_correct_tensor(predictor_setup):
    model, predictor = predictor_setup
    embedding = predictor.get_user_embedding(model, 1)

    assert embedding.shape == (2,)
    assert torch.allclose(embedding, torch.tensor([1.0, 0.0]))
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_predict.py -v
```

Expected: several failures — `TwoTowerPredictor` has no `__init__`, no `get_candidate_item_embeddings(model, ...)`, no `get_user_embedding(model, ...)`, no `invalidate_cache()`.

- [ ] **Step 3: Replace `twotower/_src/retrieval/predict.py`**

```python
from __future__ import annotations

from typing import Protocol, Sequence

import pandas as pd
import torch

from twotower._src.config import _Config


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


class TwoTowerPredictor:
    """Generate top-k recommendations from a minimal prediction interface."""

    def __init__(self) -> None:
        self._cached_all_item_embeddings: torch.Tensor | None = None
        self._cached_all_item_ids: list[int] | None = None

    def predict(
        self,
        model: _Predictable,
        *,
        user_ids: Sequence[int] | None = None,
        item_ids: Sequence[int] | None = None,
        top_k: int | None = None,
        exclude_seen: bool = True,
        strict: bool = False,
    ) -> pd.DataFrame:
        resolved_user_ids, candidate_item_ids, resolved_top_k = self.prepare_prediction_inputs(
            model,
            user_ids=user_ids,
            item_ids=item_ids,
            top_k=top_k,
            strict=strict,
        )

        empty = pd.DataFrame(columns=[model.query_col, model.candidate_col, "score", "rank"])
        if not resolved_user_ids or not candidate_item_ids:
            return empty

        model.eval()
        item_embeddings, candidate_item_ids = self.get_candidate_item_embeddings(model, candidate_item_ids)
        seen_candidates_by_query = model.get_seen_candidates_by_query() if exclude_seen else {}

        rows: list[dict[str, object]] = []
        for query_id in resolved_user_ids:
            scored_items = self.score_top_k_for_user(
                model,
                query_id=query_id,
                item_embeddings=item_embeddings,
                item_ids=candidate_item_ids,
                top_k=resolved_top_k,
                excluded_item_ids=seen_candidates_by_query.get(query_id, set()),
            )
            for rank, (candidate_id, score) in enumerate(scored_items, start=1):
                rows.append({model.query_col: query_id, model.candidate_col: candidate_id, "score": score, "rank": rank})

        return pd.DataFrame(rows, columns=[model.query_col, model.candidate_col, "score", "rank"])

    def prepare_prediction_inputs(
        self,
        model: _Predictable,
        *,
        user_ids: Sequence[int] | None,
        item_ids: Sequence[int] | None,
        top_k: int | None,
        strict: bool = False,
    ) -> tuple[list[int], list[int], int]:
        """Validate and normalize prediction inputs."""
        resolved_top_k = model.config.top_k if top_k is None else int(top_k)
        if resolved_top_k <= 0:
            raise ValueError("`top_k` must be a positive integer.")

        resolved_user_ids = (
            self._deduplicate_ids(user_ids)
            if user_ids is not None
            else model.idx_to_query_id[: min(10, len(model.idx_to_query_id))]
        )
        resolved_item_ids = (
            self._deduplicate_ids(item_ids)
            if item_ids is not None
            else list(model.idx_to_candidate_id)
        )

        unknown_user_ids = [
            query_id for query_id in resolved_user_ids if query_id not in model.query_id_to_idx
        ]
        unknown_item_ids = [
            candidate_id for candidate_id in resolved_item_ids if candidate_id not in model.candidate_id_to_idx
        ]
        if strict and (unknown_user_ids or unknown_item_ids):
            error_messages: list[str] = []
            if unknown_user_ids:
                error_messages.append(f"unknown user_ids: {unknown_user_ids[:5]}")
            if unknown_item_ids:
                error_messages.append(f"unknown item_ids: {unknown_item_ids[:5]}")
            raise ValueError("Prediction received " + "; ".join(error_messages) + ".")

        available_user_ids = [
            int(query_id) for query_id in resolved_user_ids if int(query_id) in model.query_id_to_idx
        ]
        available_item_ids = [
            int(candidate_id) for candidate_id in resolved_item_ids if int(candidate_id) in model.candidate_id_to_idx
        ]
        return available_user_ids, available_item_ids, resolved_top_k

    def score_top_k_for_user(
        self,
        model: _Predictable,
        *,
        query_id: int,
        item_embeddings: torch.Tensor,
        item_ids: list[int],
        top_k: int,
        excluded_item_ids: set[int] | None = None,
    ) -> list[tuple[int, float]]:
        if query_id not in model.query_id_to_idx:
            return []

        excluded_item_ids = excluded_item_ids or set()
        candidate_positions = [
            position for position, candidate_id in enumerate(item_ids)
            if candidate_id not in excluded_item_ids
        ]
        if not candidate_positions:
            return []

        user_embedding = self.get_user_embedding(model, query_id)
        candidate_embeddings = item_embeddings[candidate_positions]
        scores = torch.matmul(candidate_embeddings, user_embedding)

        k = min(top_k, scores.size(0))
        if k == 0:
            return []

        top_scores, top_positions = torch.topk(scores, k=k)
        return [
            (
                int(item_ids[candidate_positions[position]]),
                float(score),
            )
            for score, position in zip(top_scores.cpu().tolist(), top_positions.cpu().tolist())
        ]

    def predict_top_k_item_ids_for_user(
        self,
        model: _Predictable,
        *,
        query_id: int,
        item_embeddings: torch.Tensor,
        item_ids: list[int],
        top_k: int,
        excluded_item_ids: set[int] | None = None,
    ) -> set[int]:
        return {
            candidate_id
            for candidate_id, _score in self.score_top_k_for_user(
                model,
                query_id=query_id,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
                top_k=top_k,
                excluded_item_ids=excluded_item_ids,
            )
        }

    def get_candidate_item_embeddings(
        self,
        model: _Predictable,
        item_ids: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        """Return embeddings for `item_ids`, using the all-items cache for the full catalogue."""
        all_item_embeddings, all_item_ids = self._build_candidate_item_embeddings(model)
        if item_ids == all_item_ids:
            return all_item_embeddings, all_item_ids

        candidate_positions = [
            model.candidate_id_to_idx[candidate_id]
            for candidate_id in item_ids
            if candidate_id in model.candidate_id_to_idx
        ]
        if not candidate_positions:
            return all_item_embeddings[:0], []

        return all_item_embeddings[candidate_positions], item_ids

    def get_user_embedding(self, model: _Predictable, query_id: int) -> torch.Tensor:
        if query_id not in model.query_id_to_idx:
            raise KeyError(f"Unknown query_id: {query_id}")

        user_index = torch.tensor(
            [model.query_id_to_idx[query_id]],
            dtype=torch.long,
            device=model.device,
        )
        with torch.no_grad():
            return model.encode_queries(user_index).squeeze(0)

    def invalidate_cache(self) -> None:
        self._cached_all_item_embeddings = None
        self._cached_all_item_ids = None

    def _build_candidate_item_embeddings(self, model: _Predictable) -> tuple[torch.Tensor, list[int]]:
        if self._cached_all_item_embeddings is None or self._cached_all_item_ids is None:
            item_ids = list(model.idx_to_candidate_id)
            item_indices = torch.tensor(
                [model.candidate_id_to_idx[candidate_id] for candidate_id in item_ids],
                dtype=torch.long,
                device=model.device,
            )
            with torch.no_grad():
                item_embeddings = model.encode_candidates(item_indices)
            self._cached_all_item_embeddings = item_embeddings
            self._cached_all_item_ids = item_ids
        return self._cached_all_item_embeddings, self._cached_all_item_ids

    @staticmethod
    def _deduplicate_ids(entity_ids: Sequence[int]) -> list[int]:
        deduplicated_ids: list[int] = []
        seen_ids: set[int] = set()
        for entity_id in entity_ids:
            normalized_id = int(entity_id)
            if normalized_id in seen_ids:
                continue
            seen_ids.add(normalized_id)
            deduplicated_ids.append(normalized_id)
        return deduplicated_ids
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_predict.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add twotower/_src/retrieval/predict.py tests/test_predict.py
git commit -m "feat: move embedding cache and helpers into TwoTowerPredictor"
```

---

### Task 3: Move evaluation methods to `TwoTowerEvaluator`

`_Evaluable` currently requires `evaluate_loader`, `recall_at_k`, `popularity_recall_at_k`, `get_eval_user_ids` on the model. After this task: those four methods live on `TwoTowerEvaluator`; `_Evaluable` instead requires `encode_queries`, `encode_candidates`, `device`, `get_seen_candidates_by_query`, `get_train_positive_item_ranking`.

**Files:**
- Modify: `twotower/_src/retrieval/evaluate.py`
- Modify: `tests/test_evaluate.py`

- [ ] **Step 1: Write new/updated tests**

Replace the entire contents of `tests/test_evaluate.py` with:

```python
from __future__ import annotations

import pandas as pd
import pytest
import torch

from twotower._src.config import _Config
from twotower._src.retrieval.evaluate import EvaluateInputs, TwoTowerEvaluator


class StubEvaluableModel:
    """Stub satisfying the updated _Evaluable protocol (no evaluate_loader/recall_at_k on model)."""

    def __init__(self, evaluate_inputs: EvaluateInputs):
        self.config = _Config(top_k=10, eval_top_ks=(5, 10))
        self.device = torch.device("cpu")
        self.evaluate_inputs = evaluate_inputs
        self.query_id_to_idx = {1: 0}
        self.candidate_id_to_idx = {10: 0, 20: 1, 30: 2}
        self.idx_to_candidate_id = [10, 20, 30]
        self.ensure_fitted_calls = 0
        self.make_loader_calls: list[dict[str, object]] = []
        self.eval_calls = 0
        # user idx 0 → [1, 0], item indices 0,1,2 → [1,0],[0,1],[0,1]
        self._user_embeddings_by_idx = {0: torch.tensor([1.0, 0.0])}
        self._item_embeddings_by_idx = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([0.0, 1.0]),
            2: torch.tensor([0.0, 1.0]),
        }

    def eval(self) -> object:
        self.eval_calls += 1
        return self

    def ensure_fitted(self) -> None:
        self.ensure_fitted_calls += 1

    def build_evaluate_inputs(self, X_test: pd.DataFrame) -> EvaluateInputs:
        return self.evaluate_inputs

    def make_loader(self, *, positive_df, interactions_df, shuffle) -> list:
        self.make_loader_calls.append(
            {"positive_df": positive_df, "interactions_df": interactions_df, "shuffle": shuffle}
        )
        return []  # empty loader → evaluate_loader yields test_loss=0.0

    def resolve_eval_top_ks(self, top_k: int | None) -> list[int]:
        return [5, 10] if top_k is None else [5, int(top_k)]

    def score_pairs(self, user_input: torch.Tensor, item_input: torch.Tensor) -> torch.Tensor:
        return torch.zeros(user_input.size(0))

    def encode_queries(self, user_input: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._user_embeddings_by_idx[int(idx)] for idx in user_input])

    def encode_candidates(self, item_input: torch.Tensor) -> torch.Tensor:
        return torch.stack([self._item_embeddings_by_idx[int(idx)] for idx in item_input])

    def get_seen_candidates_by_query(self) -> dict[int, set[int]]:
        return {}

    def get_train_positive_item_ranking(self) -> list[int]:
        return []


@pytest.fixture
def evaluator_setup():
    test_input_df = pd.DataFrame({
        "event_date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "query_id": [1, 2, 3],
        "candidate_id": [10, 20, 30],
        "label": [1.0, 0.0, 1.0],
    })
    prepared_test_df = test_input_df.iloc[[0, 1]].copy()
    positive_test_df = prepared_test_df.iloc[[0]].copy()
    evaluate_inputs = EvaluateInputs(
        test_input_df=test_input_df,
        prepared_test_df=prepared_test_df,
        positive_test_df=positive_test_df,
        input_row_count=3,
        unknown_user_row_count=1,
        unknown_item_row_count=0,
    )
    return test_input_df, evaluate_inputs, TwoTowerEvaluator()


def test_evaluate_aggregates_metrics_and_uses_default_top_k(evaluator_setup):
    test_input_df, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)

    metrics = evaluator.evaluate(model, test_input_df)

    assert model.ensure_fitted_calls == 1
    assert len(model.make_loader_calls) == 1
    assert not model.make_loader_calls[0]["shuffle"]
    # evaluate_loader, recall_at_k, popularity_recall_at_k now live on TwoTowerEvaluator
    assert metrics["test_loss"] == 0.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["recall_at_10"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["popularity_recall_at_5"] == 0.0
    assert metrics["popularity_recall_at_10"] == 0.0
    assert metrics["popularity_recall_at_k"] == 0.0
    assert metrics["test_input_rows"] == 3.0
    assert metrics["test_rows_used"] == 2.0
    assert metrics["test_rows_filtered"] == 1.0
    assert metrics["test_unknown_user_rows"] == 1.0
    assert metrics["test_unknown_item_rows"] == 0.0
    assert metrics["test_positive_pairs_used_for_loss"] == 1.0
    assert metrics["test_eval_user_count"] == 1.0


def test_evaluate_respects_top_k_override(evaluator_setup):
    test_input_df, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)

    metrics = evaluator.evaluate(model, test_input_df, top_k=3)

    assert metrics["recall_at_3"] == 1.0
    assert metrics["recall_at_k"] == 1.0
    assert metrics["popularity_recall_at_3"] == 0.0
    assert metrics["popularity_recall_at_k"] == 0.0


def test_evaluate_raises_for_empty_prepared_test_set(evaluator_setup):
    test_input_df, evaluate_inputs, evaluator = evaluator_setup
    empty_model = StubEvaluableModel(EvaluateInputs(
        test_input_df=test_input_df.iloc[:0].copy(),
        prepared_test_df=evaluate_inputs.prepared_test_df.iloc[:0].copy(),
        positive_test_df=evaluate_inputs.positive_test_df.iloc[:0].copy(),
        input_row_count=0,
        unknown_user_row_count=0,
        unknown_item_row_count=0,
    ))

    with pytest.raises(RuntimeError, match="Evaluation dataset is empty"):
        evaluator.evaluate(empty_model, test_input_df)


# ── Tests for the methods that moved from TwoTower to TwoTowerEvaluator ──────


def test_get_eval_user_ids_returns_users_with_positive_labels(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    df = pd.DataFrame({
        "query_id": [1, 2, 3, 1],
        "candidate_id": [10, 20, 30, 20],
        "label": [1.0, 0.0, 1.0, 1.0],
    })

    user_ids = evaluator.get_eval_user_ids(model, df)

    assert set(user_ids) == {1, 3}


def test_get_eval_user_ids_returns_empty_for_no_positives(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    df = pd.DataFrame({"query_id": [1, 2], "candidate_id": [10, 20], "label": [0.0, 0.0]})

    assert evaluator.get_eval_user_ids(model, df) == []


def test_evaluate_loader_returns_zero_loss_for_empty_loader(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)

    metrics = evaluator.evaluate_loader(model, [], prefix="valid")

    assert metrics == {"valid_loss": 0.0}


def test_recall_at_k_returns_one_for_item_at_top(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    # prepared_test_df has query_id=1 with label=1.0 and candidate_id=10
    # model.encode_queries([0]) → [[1,0]], encode_candidates([0,1,2]) → [[1,0],[0,1],[0,1]]
    # user 1 scores: [1.0, 0.0, 0.0] → item 10 is top-1 → recall = 1.0
    recall = evaluator.recall_at_k(model, evaluate_inputs.prepared_test_df, top_k=1)

    assert recall == 1.0


def test_recall_at_k_returns_zero_for_no_positive_users(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    df = pd.DataFrame({"query_id": [1], "candidate_id": [10], "label": [0.0]})

    assert evaluator.recall_at_k(model, df, top_k=5) == 0.0


def test_popularity_recall_at_k_returns_zero_for_empty_ranking(evaluator_setup):
    _, evaluate_inputs, evaluator = evaluator_setup
    model = StubEvaluableModel(evaluate_inputs)
    # model.get_train_positive_item_ranking() → [] → popularity_recall = 0.0

    assert evaluator.popularity_recall_at_k(model, evaluate_inputs.prepared_test_df, top_k=5) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/test_evaluate.py -v
```

Expected: many failures — `TwoTowerEvaluator` missing `evaluate_loader`, `recall_at_k`, `popularity_recall_at_k`, `get_eval_user_ids`; existing tests fail because stub no longer matches `_Evaluable`.

- [ ] **Step 3: Replace `twotower/_src/retrieval/evaluate.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from twotower._src.config import _Config
from twotower._src.metrics import mean_recall, user_recall


@dataclass(slots=True)
class EvaluateInputs:
    """Prepared test artifacts required by the evaluation service."""

    test_input_df: pd.DataFrame
    prepared_test_df: pd.DataFrame
    positive_test_df: pd.DataFrame
    input_row_count: int
    unknown_user_row_count: int
    unknown_item_row_count: int


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


class TwoTowerEvaluator:
    """Evaluate a two-tower model through a minimal protocol interface."""

    def evaluate(
        self,
        model: _Evaluable,
        X_test: pd.DataFrame,
        top_k: int | None = None,
    ) -> dict[str, float]:
        model.ensure_fitted()
        evaluate_inputs = model.build_evaluate_inputs(X_test)
        if evaluate_inputs.prepared_test_df.empty:
            raise RuntimeError(
                "Evaluation dataset is empty after filtering out unknown users and items."
            )

        metrics = self.evaluate_loader(
            model,
            model.make_loader(
                positive_df=evaluate_inputs.positive_test_df,
                interactions_df=evaluate_inputs.prepared_test_df,
                shuffle=False,
            ),
            prefix="test",
        )
        for eval_top_k in model.resolve_eval_top_ks(top_k):
            metrics[f"recall_at_{eval_top_k}"] = self.recall_at_k(
                model, evaluate_inputs.prepared_test_df, eval_top_k,
            )
            metrics[f"popularity_recall_at_{eval_top_k}"] = self.popularity_recall_at_k(
                model, evaluate_inputs.prepared_test_df, eval_top_k,
            )

        selected_top_k = model.config.top_k if top_k is None else int(top_k)
        metrics["recall_at_k"] = metrics[f"recall_at_{selected_top_k}"]
        metrics["popularity_recall_at_k"] = metrics[f"popularity_recall_at_{selected_top_k}"]
        metrics["test_input_rows"] = float(evaluate_inputs.input_row_count)
        metrics["test_rows_used"] = float(len(evaluate_inputs.prepared_test_df))
        metrics["test_rows_filtered"] = float(
            evaluate_inputs.input_row_count - len(evaluate_inputs.prepared_test_df)
        )
        metrics["test_unknown_user_rows"] = float(evaluate_inputs.unknown_user_row_count)
        metrics["test_unknown_item_rows"] = float(evaluate_inputs.unknown_item_row_count)
        metrics["test_positive_rate"] = float(evaluate_inputs.prepared_test_df["label"].mean())
        metrics["test_positive_pairs_used_for_loss"] = float(len(evaluate_inputs.positive_test_df))
        metrics["test_eval_user_count"] = float(
            len(self.get_eval_user_ids(model, evaluate_inputs.prepared_test_df))
        )
        metrics["test_rows_filtered_ratio"] = (
            float(evaluate_inputs.input_row_count - len(evaluate_inputs.prepared_test_df))
            / max(float(evaluate_inputs.input_row_count), 1.0)
        )
        return metrics

    def evaluate_loader(
        self,
        model: _Evaluable,
        loader: DataLoader[Any],
        prefix: str = "valid",
    ) -> dict[str, float]:
        model.eval()
        criterion = nn.LogSigmoid()
        loss_sum = 0.0
        total = 0

        with torch.no_grad():
            for user_batch, pos_item_batch, neg_item_batch in loader:
                user_batch = user_batch.to(model.device)
                pos_item_batch = pos_item_batch.to(model.device)
                neg_item_batch = neg_item_batch.to(model.device)

                positive_scores = model.score_pairs(user_batch, pos_item_batch)
                negative_scores = model.score_pairs(user_batch, neg_item_batch)
                loss: torch.Tensor = -criterion(positive_scores - negative_scores).mean()

                batch_size = user_batch.size(0)
                loss_sum += loss.item() * batch_size
                total += batch_size

        return {f"{prefix}_loss": loss_sum / max(total, 1)}

    def get_eval_user_ids(self, model: _Evaluable, evaluation_df: pd.DataFrame) -> list[int]:
        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        if positive_df.empty:
            return []
        return list(
            positive_df["query_id"]
            .drop_duplicates()
            .head(model.config.max_eval_users)
            .astype(int)
        )

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

    def popularity_recall_at_k(
        self,
        model: _Evaluable,
        evaluation_df: pd.DataFrame,
        top_k: int,
    ) -> float:
        candidate_user_ids = self.get_eval_user_ids(model, evaluation_df)
        if not candidate_user_ids:
            return 0.0

        popularity_ranking = model.get_train_positive_item_ranking()
        if not popularity_ranking:
            return 0.0

        positive_df = evaluation_df[evaluation_df["label"] == 1.0]
        seen_candidates_by_query = model.get_seen_candidates_by_query()
        recalls = []
        for query_id in candidate_user_ids:
            actual_items = set(
                positive_df.loc[positive_df["query_id"] == query_id, "candidate_id"].astype(int)
            )
            if not actual_items:
                continue

            excluded_item_ids = seen_candidates_by_query.get(query_id, set())
            predicted_items: list[int] = []
            for candidate_id in popularity_ranking:
                if candidate_id in excluded_item_ids:
                    continue
                predicted_items.append(candidate_id)
                if len(predicted_items) == top_k:
                    break

            recalls.append(user_recall(actual_items, set(predicted_items)))

        return mean_recall(recalls)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/test_evaluate.py -v
```

Expected: all tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add twotower/_src/retrieval/evaluate.py tests/test_evaluate.py
git commit -m "feat: move recall, popularity_recall, evaluate_loader, get_eval_user_ids to TwoTowerEvaluator"
```

---

### Task 4: Slim `core.py` — remove moved methods, add delegators

Now that the services own the logic, strip `TwoTower` of the redundant implementations. `recall_at_k` and `invalidate_item_embedding_cache` become thin delegators.

**Files:**
- Modify: `twotower/_src/core.py`

- [ ] **Step 1: Remove moved state and methods from `TwoTower.__init__`**

In [twotower/_src/core.py](twotower/_src/core.py), find and delete these two lines from `__init__`:

```python
        self._cached_all_item_embeddings: torch.Tensor | None = None
        self._cached_all_item_ids: list[int] | None = None
```

- [ ] **Step 2: Remove moved methods from `TwoTower`**

Delete the following method bodies (keep the delegators in step 3):

- `evaluate_loader` (lines ~368–392)
- `get_eval_user_ids` (lines ~407–416)
- `recall_at_k` (the full implementation, lines ~418–440) — replaced below
- `popularity_recall_at_k` (lines ~442–470)
- `get_candidate_item_embeddings` (lines ~504–520)
- `get_user_embedding` (lines ~522–533)
- `_build_candidate_item_embeddings` (lines ~647–661)

- [ ] **Step 3: Replace `recall_at_k` and `invalidate_item_embedding_cache` with delegators**

Replace the deleted `recall_at_k` with:

```python
    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float:
        return self._evaluator.recall_at_k(self, evaluation_df, top_k, exclude_seen=exclude_seen)
```

Replace the deleted `invalidate_item_embedding_cache` body with:

```python
    def invalidate_item_embedding_cache(self) -> None:
        self._predictor.invalidate_cache()
```

- [ ] **Step 4: Remove unused imports from `core.py`**

Remove these import lines that are no longer used after the cleanup:

```python
import torch.nn as nn
import torch.nn.functional as F
from twotower._src.metrics import mean_recall, user_recall
```

Verify the remaining `import torch` and `from torch.utils.data import DataLoader` are still used (`torch` is used in `resolve_device` and `__init__`; `DataLoader` is used in `make_loader`).

- [ ] **Step 5: Run the full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all existing tests PASSED, no regressions. Count the lines in core.py:

```bash
wc -l twotower/_src/core.py
```

- [ ] **Step 6: Commit**

```bash
git add twotower/_src/core.py
git commit -m "refactor: slim TwoTower by removing moved methods; add delegators for recall_at_k and invalidate_cache"
```

---

### Task 5: Compose `_Trainable` from sub-protocols in `fit.py`

`_Trainable` currently re-declares attributes and methods already captured by the shared protocols. Make it inherit from them, removing duplication.

**Files:**
- Modify: `twotower/_src/training/fit.py`

- [ ] **Step 1: Import the shared protocols**

In [twotower/_src/training/fit.py](twotower/_src/training/fit.py), add this import after the existing imports:

```python
from twotower._src.protocols import _HasConfig, _HasEmbeddings, _HasIDMappings
```

- [ ] **Step 2: Replace `_Trainable` with the composed version**

Find the existing `_Trainable` class definition and replace it with:

```python
class _Trainable(_HasEmbeddings, _HasIDMappings, _HasConfig, Protocol):
    """Minimal model contract required by the training module."""

    def build_towers(self, num_users: int, num_items: int) -> None: ...

    def to(self, device: torch.device) -> nn.Module: ...

    def parameters(self) -> Iterator[nn.Parameter]: ...

    def state_dict(self) -> dict[str, torch.Tensor]: ...

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None: ...

    def train(self, mode: bool = True) -> object: ...

    def eval(self) -> object: ...

    def recall_at_k(self, evaluation_df: pd.DataFrame, top_k: int, exclude_seen: bool = True) -> float: ...
```

The sub-protocols cover `encode_queries`, `encode_candidates`, `score_pairs`, `retrieval_logits`, all four ID mappings, and `config`. `_Trainable` adds only what's training-specific on top.

- [ ] **Step 3: Run the full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all tests PASSED — the refactor is purely structural (protocol inheritance doesn't change runtime behaviour).

- [ ] **Step 4: Commit**

```bash
git add twotower/_src/training/fit.py
git commit -m "refactor: compose _Trainable from _HasEmbeddings, _HasIDMappings, _HasConfig"
```
