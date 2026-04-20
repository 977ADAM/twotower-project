from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.сonfig import Config
from twotower import TwoTower

app = FastAPI(
    title="Recsys API",
    description="Recommender system API",
    version="1.0.0",
)


class RecommendationRequest(BaseModel):
    user_ids: list[int] | None = None
    item_ids: list[int] | None = None
    top_k: int | None = Field(default=None, ge=1)
    exclude_seen: bool = True
    strict: bool = False


@lru_cache(maxsize=1)
def get_model() -> TwoTower:
    checkpoint_path = Path(Config().model_save_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint was not found: {checkpoint_path}")
    return TwoTower().load_model(checkpoint_path)


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    checkpoint_path = Path(Config().model_save_path)
    return {
        "status": "ok",
        "model_path": str(checkpoint_path),
        "model_exists": checkpoint_path.exists(),
    }


@app.post("/recommendations")
def recommend(request: RecommendationRequest):
    try:
        predictions = get_model().predict(
            user_ids=request.user_ids,
            item_ids=request.item_ids,
            top_k=request.top_k,
            exclude_seen=request.exclude_seen,
            strict=request.strict,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"predictions": predictions}
