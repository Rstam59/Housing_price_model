# import logging
# import os

# def setup_logging(level: str | None = None) -> None:
#     lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
#     logging.basicConfig(
#         level=getattr(logging, lvl, logging.INFO),
#         format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
#     )

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Attach structured extras if present
        # We’ll pass extras like: extra={"request_id": "...", "path": "..."}
        for k in ("request_id", "path", "method", "status_code", "latency_ms", "rows", "client_ip"):
            if hasattr(record, k):
                base[k] = getattr(record, k)

        if record.exc_info:
            base["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(base, ensure_ascii=False)


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_format = (os.getenv("LOG_FORMAT", "pretty") or "pretty").lower()

    root = logging.getLogger()
    root.setLevel(getattr(logging, lvl, logging.INFO))

    # Remove existing handlers (important for uvicorn reload / notebooks)
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))

    root.addHandler(handler)

    # Reduce noisy loggers if needed
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
