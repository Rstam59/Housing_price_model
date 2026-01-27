import hashlib 
from pathlib import Path
import json 
from typing import Any, Dict

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    p = Path(path)
    return sha256_bytes(p.read_bytes())


def sha256_json(obj: Dict[str, Any]) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return sha256_bytes(s.encode('utf-8'))


def short_hash(h: str, n: int = 12) -> str:
    return h[:n]

