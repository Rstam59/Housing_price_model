from __future__ import annotations
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .predictor import load_predictor
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

    model_path = serve_cfg["artifacts"]["model_path"]
    profile_path = serve_cfg["artifacts"].get("training_profile_path")

    pred = load_predictor(model_path=model_path, training_profile_path=profile_path)

    pred.allow_extra_columns = bool(serve_cfg["behavior"].get("allow_extra_columns", False))
    pred.strict_categories = bool(serve_cfg["behavior"].get("strict_categories", False))

    app.state.serve_cfg = serve_cfg
    app.state.predictor = pred

    logger.info("Service started. model=%s profile=%s profile_loaded=%s",
                model_path, profile_path, "yes" if pred.training_profile else "no")

    yield


    logger.info("Service shutting down.")


app = FastAPI(title="Housing Price Model", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/meta")
def meta(request: Request):
    serve_cfg = request.app.state.serve_cfg
    pred = request.app.state.predictor
    return {
        "model_path": serve_cfg["artifacts"]["model_path"],
        "training_profile_path": serve_cfg["artifacts"].get("training_profile_path"),
        "training_profile_loaded": bool(pred.training_profile),
        "allow_extra_columns": pred.allow_extra_columns,
        "strict_categories": pred.strict_categories,
        "required_columns": REQUIRED_COLUMNS,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request):
    pred = request.app.state.predictor

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
        raise HTTPException(status_code=422, detail={"error": str(e), "details": e.details})
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal error")

    latency_ms = (time.time() - t0) * 1000.0
    logger.info("predict_ok rows=%d latency_ms=%.2f", len(req.records), latency_ms)

    return PredictResponse(
        predictions=result["predictions"],
        drift=result.get("drift"),
        latency_ms=latency_ms,
    )
