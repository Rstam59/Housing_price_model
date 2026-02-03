from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .middleware import PayloadLimitMiddleware, RequestIdMiddleware
from .predictor import load_predictor
from .registry import ModelRegistry
from .schema import REQUIRED_COLUMNS, SchemaError, validate_dataframe

logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    predictions: List[float]
    drift: Optional[Dict[str, Any]] = None
    latency_ms: float


def _load_serve_cfg() -> dict:
    return yaml.safe_load(Path("configs/serve.yaml").read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    serve_cfg = _load_serve_cfg()

    registry_dir = Path(serve_cfg["artifacts"]["registry_dir"])
    profile_path = serve_cfg["artifacts"].get("training_profile_path")

    registry = ModelRegistry(registry_dir)

    pred = None
    model_load_error: Optional[str] = None
    active_model_path: Optional[str] = None

    try:
        active_model_path = str(registry.resolve_active())
        pred = load_predictor(model_path=active_model_path, training_profile_path=profile_path)
        pred.allow_extra_columns = bool(serve_cfg["behavior"].get("allow_extra_columns", False))
        pred.strict_categories = bool(serve_cfg["behavior"].get("strict_categories", False))

        logger.info(
            "model_loaded",
            extra={
                "path": "startup",
                "request_id": "system",
                "status_code": 200,
                "model_path": active_model_path,
                "profile_loaded": bool(pred.training_profile),
            },
        )
    except FileNotFoundError as e:
        # This is NORMAL in CI / fresh deploy.
        model_load_error = str(e)
        logger.warning(
            "model_not_loaded",
            extra={
                "path": "startup",
                "request_id": "system",
                "status_code": 503,
                "error": model_load_error,
            },
        )
    except Exception as e:
        model_load_error = repr(e)
        logger.exception(
            "model_load_failed",
            extra={
                "path": "startup",
                "request_id": "system",
                "status_code": 500,
            },
        )

    # store state
    app.state.serve_cfg = serve_cfg
    app.state.registry = registry
    app.state.predictor = pred
    app.state.model_load_error = model_load_error

    yield

    logger.info("service_shutdown", extra={"path": "shutdown", "request_id": "system"})


app = FastAPI(title="Housing Price Model", version="0.1.0", lifespan=lifespan)

# Middlewares: keep them always on; they don't depend on model existence.
# Use safe defaults here; we'll load real limits from app.state in the middleware init below.
app.add_middleware(RequestIdMiddleware)

# IMPORTANT:
# Starlette middleware args are evaluated at app startup (import time).
# To avoid reading configs at import time, we choose conservative defaults here.
# You can tighten them later if you want, but this already prevents abuse.
app.add_middleware(
    PayloadLimitMiddleware,
    max_body_bytes=1_000_000,
    max_records=512,
    path_prefix="/predict",
)


@app.get("/health")
def health(request: Request):
    # Health should not mean "model is loaded". It means "process is up".
    # You can add "model_loaded" flag so infra can decide.
    pred = request.app.state.predictor
    return {"status": "ok", "model_loaded": pred is not None}


@app.get("/meta")
def meta(request: Request):
    serve_cfg = request.app.state.serve_cfg
    pred = request.app.state.predictor
    registry = request.app.state.registry

    try:
        active_model = str(registry.resolve_active())
    except FileNotFoundError:
        active_model = None

    return {
        "active_model_path": active_model,
        "training_profile_loaded": bool(pred.training_profile) if pred else False,
        "allow_extra_columns": bool(getattr(pred, "allow_extra_columns", False)) if pred else False,
        "strict_categories": bool(getattr(pred, "strict_categories", False)) if pred else False,
        "required_columns": REQUIRED_COLUMNS,
        "limits": serve_cfg.get("limits", {}),
        "model_load_error": request.app.state.model_load_error,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    pred = request.app.state.predictor
    rid = getattr(request.state, "request_id", "unknown")

    # If model isn't loaded, return 503 instead of crashing the whole service.
    if pred is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "Model not loaded", "reason": request.app.state.model_load_error},
        )

    t0 = time.time()
    try:
        df = pd.DataFrame(req.records)

        df = validate_dataframe(
            df,
            allow_extra_columns=pred.allow_extra_columns,
            strict_categories=pred.strict_categories,
            require_non_empty=True,
        )
        df = df[REQUIRED_COLUMNS]

        result = pred.predict_df(df)

    except SchemaError as e:
        logger.warning(
            "predict_schema_error",
            extra={"request_id": rid, "path": "/predict", "rows": len(req.records), "status_code": 422},
        )
        raise HTTPException(status_code=422, detail={"error": str(e), "details": e.details})

    except Exception:
        logger.exception("predict_failed", extra={"request_id": rid, "path": "/predict", "status_code": 500})
        raise HTTPException(status_code=500, detail="Internal error")

    latency_ms = (time.time() - t0) * 1000.0
    logger.info(
        "predict_ok",
        extra={
            "request_id": rid,
            "path": "/predict",
            "method": "POST",
            "rows": len(req.records),
            "latency_ms": latency_ms,
            "status_code": 200,
        },
    )

    return PredictResponse(
        predictions=result["predictions"],
        drift=result.get("drift"),
        latency_ms=latency_ms,
    )
