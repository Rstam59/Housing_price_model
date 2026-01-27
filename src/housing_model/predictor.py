from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from .profiling import compare_to_profile
from .schema import REQUIRED_COLUMNS


@dataclass
class Predictor:
    model: Any
    training_profile: Optional[Dict[str, Any]] = None

    def predict_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Final safety: enforce exact schema + column order
        x = df[REQUIRED_COLUMNS]

        preds = self.model.predict(x)
        preds = [float(p) for p in preds]  # JSON-friendly

        drift = None
        if self.training_profile is not None:
            drift = compare_to_profile(x, self.training_profile)

        return {"predictions": preds, "drift": drift}


def load_predictor(model_path: str, training_profile_path: str | None = None) -> Predictor:
    model = joblib.load(model_path)

    profile = None
    if training_profile_path:
        import json
        profile = json.loads(Path(training_profile_path).read_text(encoding="utf-8"))

    return Predictor(model=model, training_profile=profile)
