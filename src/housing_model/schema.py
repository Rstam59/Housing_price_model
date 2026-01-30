from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple 
import pandas as pd 

REQUIRED_NUMERIC = [
    'longitude',
    'latitude',
    'housing_median_age',
    'total_rooms',
    'total_bedrooms',
    'population',
    'households',
    'median_income'
]


REQUIRED_CATEGORICAL = ['ocean_proximity']

REQUIRED_COLUMNS = REQUIRED_NUMERIC + REQUIRED_CATEGORICAL

ALLOWED_OCEAN_PROXIMITY = [
    '<1H OCEAN',
    'INLAND',
    'ISLAND',
    'NEAR BAY',
    'NEAR OCEAN'
]

@dataclass(frozen=True)
class SchemaError(Exception):
    message: str
    details: Dict[str, Any] | None = None 

    def __str__(self) -> str:
        if not self.details:
            return self.message
        return f'{self.message} | details = {self.details}'
    

def _missing_and_extra_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    cols = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    extra = [c for c in df.columns if c not in set(REQUIRED_COLUMNS)]
    return missing, extra


def validate_dataframe(
        df: pd.DataFrame,
        *,
        allow_extra_columns: bool = False,
        strict_categories: bool = False,
        require_non_empty: bool = True 
) -> pd.DataFrame:
    if require_non_empty and len(df) == 0:
        raise SchemaError('Input dataframe is empty.')
    
    missing, extra = _missing_and_extra_columns(df)

    if missing:
        raise SchemaError('Missing required columns.', {'missing': missing})
    
    if (not allow_extra_columns) and extra:
        raise SchemaError('Unexpected extra columns.', {'extra': extra})
    
    out = df.copy()

    for c in REQUIRED_NUMERIC: 
        out[c] = pd.to_numeric(out[c], errors='coerce')

    bad_numeric = []
    for c in REQUIRED_NUMERIC:
        if out[c].isna().all():
            bad_numeric.append(c)
    if bad_numeric:
        raise SchemaError('Numeric columns contaion no valid numbers', {'all_nan_columns': bad_numeric})
    
    
    if strict_categories:
        unknown = sorted(set(out['ocean_proximity'].dropna().unique()) - ALLOWED_OCEAN_PROXIMITY)
        if unknown:
            raise SchemaError('Unknown ocean_proximity category.', {'unknown': unknown})
    
    return out 