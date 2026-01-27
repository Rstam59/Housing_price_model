from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import sklearn
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, field_validator

from .predictor import load_predictor, Predictor
from .schema import REQUIRED_COLUMNS, SchemaError, validate_dataframe

logger = logging.getLogger(__name__)


# -------------------------
# serve.yaml config model
# -------------------------
@dataclass(frozen=True)
class ServeBehavior:
    allow_extra_columns: bool = False
    strict_categories: bool = False
    return_drift_in_response: bool = True
    max_rows_per_request: int = 1000


@dataclass(frozen=True)
class ServeArtifacts:
    model_path: str
    training_profile_path: Optional[str] = None


@dataclass(frozen=True)
class ServeConfig:
    artifacts: ServeArtifacts
    behavior: ServeBehavior


def load_serve_config(path: str = "configs/serve.yaml") -> ServeConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    art = payload.get("artifacts", {}) or {}
    beh = payload.get("behavior", {}) or {}

    model_path = art.get("model_path")
    if not model_path:
        raise ValueError("configs/serve.yaml missing artifacts.model_path")

    return ServeConfig(
        artifacts=ServeArtifacts(
            model_path=str(model_path),
            training_profile_path=str(art["training_profile_path"]) if art.get("training_profile_path") else None,
        ),
        behavior=ServeBehavior(
            allow_extra_columns=bool(beh.get("allow_extra_columns", False)),
            strict_categories=bool(beh.get("strict_categories", False)),
            return_drift_in_response=bool(beh.get("return_drift_in_response", True)),
            max_rows_per_request=int(beh.get("max_rows_per_request", 1000)),
        ),
    )


# -------------------------
# API models
# -------------------------
class PredictRequest(BaseModel):
    # Accept list of record dicts.
    # Convenience: also accept a single dict and coerce to list.
    records: List[Dict[str, Any]] = Field(..., min_length=1)

    @field_validator("records", mode="before")
    @classmethod
    def _coerce_single_record(cls, v):
        if isinstance(v, dict):
            return [v]
        return v


class PredictResponse(BaseModel):
    predictions: List[float]
    drift: Optional[Dict[str, Any]] = None
    latency_ms: float
    model_version: str


# -------------------------
# App
# -------------------------
def create_app() -> FastAPI:
    app = FastAPI(title="Housing Price Model", version="0.1.0")

    @app.on_event("startup")
    def startup() -> None:
        cfg = load_serve_config("configs/serve.yaml")

        pred = load_predictor(
            model_path=cfg.artifacts.model_path,
            training_profile_path=cfg.artifacts.training_profile_path,
        )

        app.state.cfg = cfg
        app.state.pred = pred

        logger.info(
            "service_start %s",
            json.dumps(
                {
                    "event": "service_start",
                    "model_path": cfg.artifacts.model_path,
                    "profile_path": cfg.artifacts.training_profile_path,
                    "sklearn": sklearn.__version__,
                }
            ),
        )

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/meta")
    def meta(request: Request) -> Dict[str, Any]:
        cfg: ServeConfig = request.app.state.cfg
        return {
            "service_version": request.app.version,
            "sklearn": sklearn.__version__,
            "model_path": cfg.artifacts.model_path,
            "training_profile_path": cfg.artifacts.training_profile_path,
            "behavior": {
                "allow_extra_columns": cfg.behavior.allow_extra_columns,
                "strict_categories": cfg.behavior.strict_categories,
                "return_drift_in_response": cfg.behavior.return_drift_in_response,
                "max_rows_per_request": cfg.behavior.max_rows_per_request,
            },
            "required_columns": REQUIRED_COLUMNS,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest, request: Request) -> PredictResponse:
        cfg: ServeConfig = request.app.state.cfg
        pred: Predictor = request.app.state.pred

        # Basic DoS / misuse protection
        if len(req.records) > cfg.behavior.max_rows_per_request:
            raise HTTPException(
                status_code=413,
                detail=f"Too many rows. max_rows_per_request={cfg.behavior.max_rows_per_request}",
            )

        t0 = time.time()

        try:
            df = pd.DataFrame(req.records)

            # Validate + coerce types according to schema policy
            df = validate_dataframe(
                df,
                allow_extra_columns=cfg.behavior.allow_extra_columns,
                strict_categories=cfg.behavior.strict_categories,
                require_non_empty=True,
            )

            # Enforce exact contract + stable order
            df = df[REQUIRED_COLUMNS]

            result = pred.predict_df(df)

        except SchemaError as e:
            raise HTTPException(status_code=422, detail={"error": str(e), "details": e.details})
        except Exception:
            logger.exception("predict_failed")
            raise HTTPException(status_code=500, detail="Internal error")

        latency_ms = (time.time() - t0) * 1000.0

        drift = result.get("drift")
        drift_out = drift if cfg.behavior.return_drift_in_response else None

        logger.info(
            "%s",
            json.dumps(
                {
                    "event": "predict_ok",
                    "rows": len(req.records),
                    "latency_ms": round(latency_ms, 2),
                    "drift": bool(drift),
                }
            ),
        )

        return PredictResponse(
            predictions=result["predictions"],
            drift=drift_out,
            latency_ms=latency_ms,
            model_version=request.app.version,
        )

    return app


# Uvicorn entrypoint: uvicorn housing_model.service:app --reload
app = create_app()
