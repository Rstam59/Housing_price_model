from __future__ import annotations

import json
import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

# Two separate concerns in one file:
# - Request ID injection (for tracing logs)
# - Payload limits (body size + max records)
#
# Mental model:
# Every request must be traceable (request_id).
# Every request must be bounded (size + complexity) or you get DoS'd.

REQUEST_ID_HEADER = "X-Request-Id"


class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER)
        if not rid:
            rid = uuid.uuid4().hex

        request.state.request_id = rid

        t0 = time.time()
        response = await call_next(request)
        latency_ms = (time.time() - t0) * 1000.0

        response.headers[REQUEST_ID_HEADER] = rid
        response.headers["X-Latency-Ms"] = f"{latency_ms:.2f}"
        return response


class PayloadLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        *,
        max_body_bytes: int = 1_000_000,   # ~1MB
        max_records: int = 512,
        path_prefix: str = "/predict",
    ):
        super().__init__(app)
        self.max_body_bytes = int(max_body_bytes)
        self.max_records = int(max_records)
        self.path_prefix = path_prefix

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only enforce on predict endpoints (avoid breaking /health etc.)
        if not request.url.path.startswith(self.path_prefix):
            return await call_next(request)

        # Content-Length check (fast reject when present)
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_body_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"error": "Payload too large", "max_body_bytes": self.max_body_bytes},
                    )
            except ValueError:
                # ignore bad header, fall back to reading body
                pass

        # Read body (Starlette caches it; downstream can still call request.json()).
        body = await request.body()
        if len(body) > self.max_body_bytes:
            return JSONResponse(
                status_code=413,
                content={"error": "Payload too large", "max_body_bytes": self.max_body_bytes},
            )

        # Lightweight semantic limit: number of records
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception:
            return JSONResponse(status_code=400, content={"error": "Invalid JSON body"})

        records = payload.get("records")
        if not isinstance(records, list):
            return JSONResponse(status_code=422, content={"error": "Body must contain 'records' as a list"})

        if len(records) > self.max_records:
            return JSONResponse(
                status_code=413,
                content={"error": "Too many records", "max_records": self.max_records, "received": len(records)},
            )

        return await call_next(request)
