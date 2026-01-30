from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import joblib
import pandas as pd

from .schema import validate_dataframe, REQUIRED_COLUMNS
from .profiling import compare_to_profile

@dataclass
class Predictor:
    model: Any
    training_profile: Optional[Dict[str, Any]] = None
    allow_extra_columns: bool = False
    strict_categories: bool = False

    def predict_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        x = validate_dataframe(
            df,
            allow_extra_columns=self.allow_extra_columns,
            strict_categories=self.strict_categories,
            require_non_empty=True,
        )
        x = x[REQUIRED_COLUMNS]  # enforce exact contract for sklearn pipeline

        preds = self.model.predict(x)
        preds = [float(p) for p in preds]

        drift = None
        if self.training_profile is not None:
            drift = compare_to_profile(x, self.training_profile)

        return {"predictions": preds, "drift": drift}


def _load_json_optional(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read().strip()
        if not txt:
            return None
        return json.loads(txt)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        # Corrupt/partial file: treat as absent, don’t take down service
        return None


def load_predictor(model_path: str, training_profile_path: str | None = None) -> Predictor:
    model = joblib.load(model_path)
    profile = _load_json_optional(training_profile_path) if training_profile_path else None
    return Predictor(model=model, training_profile=profile)
