from typing import Any, Dict 
import pandas as pd 

from .schema import REQUIRED_NUMERIC, REQUIRED_CATEGORICAL

def build_training_profile(df: pd.DataFrame) -> Dict[str, Any]:
    profile: Dict[str, Any] = {'numeric': {}, 'categorical': {}}

    for col in REQUIRED_NUMERIC:
        s = df[col] 
        profile['numeric'][col] = {
            'count': int(s.notna().sum()),
            'q01': float(s.quantile(0.01)),
            'q05': float(s.quantile(0.05)),
            'q50': float(s.quantile(0.50)),
            'q95': float(s.quantile(0.95)),
            'q99': float(s.quantile(0.99)),
        }

    for col in REQUIRED_CATEGORICAL:
        freq = df[col].fillna("<<MISSING>>").value_counts(normalize=True)
        profile['categorical'][col] = {k: float(v) for k, v in freq.to_dict().items()}
    return profile


def compare_to_profile(df: pd.DataFrame, profile: Dict[str, Any]) -> Dict[str, Any]:

    out: Dict[str, Any] = {'numeric': {}, 'categorical': {}}

    for col, base in profile.get('numeric', {}).items():
        cur_med = float(df[col].quantile(0.50))
        base_med = float(base['q50'])
        denom = abs(base_med) if abs(base_med) > 1e-9 else 1.0
        rel_shift = (cur_med - base_med) / denom
        out["numeric"][col] = {"current_q50": cur_med, "baseline_q50": base_med, "rel_shift": float(rel_shift)}


    
    for col, base_freq in profile.get('categorical', {}).items():
        cur = df[col].fillna('<<MISSING>>').value_counts(normalize = True).to_dict()
        unknown_share = 0.0
        for k, v in cur.items():
            if k not in base_freq:
                unknown_share += float(v)
        out['categorical'][col] = {'unknown_share': float(unknown_share), 'current_top': sorted(cur.items(), key=lambda x: -x[1])[:5]}
    

    return out
    