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


def test_get_candidate_item_embeddings_subset_filters_unknown_ids(predictor_setup):
    model, predictor = predictor_setup
    # Request [10, 999] — 999 is unknown; should return only embedding for 10
    embeddings, returned_ids = predictor.get_candidate_item_embeddings(model, [10, 999])

    assert returned_ids == [10]
    assert embeddings.shape == (1, 2)
    assert torch.allclose(embeddings[0], torch.tensor([1.0, 0.0]))


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
