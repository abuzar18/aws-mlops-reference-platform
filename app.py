from dataclasses import asdict

from fastapi import FastAPI
from pydantic import BaseModel, Field

from platform_core import ReleaseMetrics, deterministic_score, evaluate_release

app = FastAPI(title="AWS MLOps Reference Platform", version="1.0.0")


class PredictionPayload(BaseModel):
    features: list[float] = Field(min_length=1, max_length=100)


class ReleasePayload(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    accuracy: float = Field(ge=0, le=1)
    p95_latency_ms: float = Field(gt=0)
    error_rate: float = Field(ge=0, le=1)
    drift_psi: float = Field(ge=0)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "model_version": "reference-1.0.0"}


@app.post("/predict")
def predict(payload: PredictionPayload) -> dict[str, float | str]:
    return {
        "score": deterministic_score(payload.features),
        "model_version": "reference-1.0.0",
    }


@app.post("/releases/evaluate")
def evaluate(payload: ReleasePayload) -> dict:
    metrics = ReleaseMetrics(**payload.model_dump())
    return asdict(evaluate_release(metrics))
