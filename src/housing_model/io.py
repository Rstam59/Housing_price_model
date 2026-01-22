# with open("artifacts/metrics.json", "w") as f:
#     json.dump(metrics, f)

import json 
from pathlib import Path 


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str, payload: dict) -> str: 
    ensure_parent(path)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return path 