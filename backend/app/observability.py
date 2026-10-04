"""Structured logging and request correlation.

Deliberately small: stdlib logging with a JSON formatter and a request-id
context var. No OpenTelemetry, no Prometheus, no log shipper. One process, one
operator, `docker logs`-shaped output.
"""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

REQUEST_ID: ContextVar[str] = ContextVar("request_id", default="-")

# stdlib LogRecord attributes, so `extra=` fields can be separated from them
_STD = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
    "asctime", "message", "taskName"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
                  + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": REQUEST_ID.get(),
        }
        payload.update({k: v for k, v in record.__dict__.items() if k not in _STD})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # uvicorn duplicates access lines in its own format; let ours be the record
    logging.getLogger("uvicorn.access").handlers[:] = []
    logging.getLogger("uvicorn.access").propagate = False


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, logs one line per request with its latency, and
    echoes the id in `X-Request-ID` so a user-reported error can be found in the
    logs without guessing."""

    def __init__(self, app, logger_name: str = "api.access"):
        super().__init__(app)
        self.log = logging.getLogger(logger_name)

    async def dispatch(self, request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        token = REQUEST_ID.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            self.log.exception(
                "unhandled exception",
                extra={"method": request.method, "path": request.url.path,
                       "duration_ms": round((time.perf_counter() - started) * 1000, 1)})
            # The 500 is built HERE rather than by app.errors' catch-all, which
            # FastAPI installs inside Starlette's ServerErrorMiddleware --
            # outside this middleware, and therefore after `finally` has reset
            # the context var. This is the last layer that still knows `rid`, and
            # a 500 is the one response where the correlation id is the whole
            # point of the message.
            return JSONResponse(
                status_code=500,
                content={"detail": f"Internal error. Quote request_id {rid} "
                                    f"when reporting this.",
                         "error": "internal_error", "request_id": rid},
                headers={"X-Request-ID": rid},
            )
        else:
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            response.headers["X-Request-ID"] = rid
            self.log.info(
                "request",
                extra={"method": request.method, "path": request.url.path,
                       "status": response.status_code, "duration_ms": duration_ms})
            return response
        finally:
            # Reset LAST: the access-log call above reads the context var, so
            # resetting before it logs request_id "-" on every line.
            REQUEST_ID.reset(token)
