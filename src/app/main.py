from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

app = FastAPI(
    title="Recsys API",
    description="Recommender system API",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "ok"}



class RecommendationRequest(BaseModel):
    
    