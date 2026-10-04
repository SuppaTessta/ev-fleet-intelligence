"""Exception handling for the API.

Three rules:

- Domain errors map to 4xx and carry their own message, which is written for
  the caller.
- A missing or unreadable model artifact maps to 503, not 4xx. The artifacts are
  gitignored, so a fresh deploy has none, and reporting that as "you sent bad
  data" keeps 5xx alerting green while the service cannot answer.
- Anything unexpected maps to a generic 500 whose body is a request id. The
  exception text never reaches the caller: `str(e)` on a FileNotFoundError is an
  absolute filesystem path. The traceback goes to the log under that id.
"""

import logging
import math

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.observability import REQUEST_ID

log = logging.getLogger("api.error")


class ModelUnavailableError(RuntimeError):
    """A required model artifact is missing or unreadable. Maps to 503 -- the
    service cannot answer right now, and it is not the caller's fault."""


class DomainValidationError(ValueError):
    """Input is well-formed but semantically invalid (unknown vehicle model,
    unknown priority tier). Maps to 422; the message is safe to return."""


def _body(status: int, detail: str, kind: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"detail": detail, "error": kind, "request_id": REQUEST_ID.get()},
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(DomainValidationError)
    async def _domain(request: Request, exc: DomainValidationError):
        log.warning("domain validation failed", extra={"path": request.url.path,
                                                        "detail": str(exc)})
        return _body(422, str(exc), "domain_validation_error")

    @app.exception_handler(ModelUnavailableError)
    async def _unavailable(request: Request, exc: ModelUnavailableError):
        # error level: an operator problem, not a caller problem
        log.error("model artifact unavailable", extra={"path": request.url.path,
                                                        "detail": str(exc)})
        return _body(503, "A required model artifact is unavailable. See /ready.",
                     "model_unavailable")

    @app.exception_handler(FileNotFoundError)
    async def _missing_file(request: Request, exc: FileNotFoundError):
        # the path in `exc` is deliberately NOT returned to the caller
        log.error("required file missing", exc_info=exc, extra={"path": request.url.path})
        return _body(503, "A required data file is unavailable. See /ready.",
                     "model_unavailable")

    @app.exception_handler(RequestValidationError)
    async def _request_validation(request: Request, exc: RequestValidationError):
        # exc.errors() echoes the offending input back. Two values have to be
        # rewritten before that is safe to serialise or send:
        #   - non-finite floats cannot be JSON-encoded, so returning them raw
        #     makes the error response itself fail with "Out of range float
        #     values are not JSON compliant" -- and rejecting them is the point.
        #   - a large collection makes the error body bigger than the request
        #     that caused it.
        safe = []
        for err in exc.errors():
            item = {k: v for k, v in err.items() if k != "ctx"}
            value = item.get("input")
            if isinstance(value, float) and not math.isfinite(value):
                item["input"] = repr(value)
            elif isinstance(value, (list, dict)) and len(value) > 5:
                item["input"] = (f"<{type(value).__name__} of {len(value)} items, "
                                 f"omitted from this error>")
            item["loc"] = [str(part) for part in item.get("loc", ())]
            safe.append(item)
        log.info("request validation failed",
                 extra={"path": request.url.path, "errors": safe})
        return JSONResponse(
            status_code=422,
            content={"detail": safe, "error": "request_validation_error",
                     "request_id": REQUEST_ID.get()},
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException):
        return _body(exc.status_code, str(exc.detail), "http_error")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # Backstop only. RequestContextMiddleware builds the 500 for anything
        # raised inside the middleware stack, because this handler runs in
        # Starlette's ServerErrorMiddleware -- outside it, and after the request
        # id context var has been reset.
        log.exception("unhandled exception", extra={"path": request.url.path})
        return _body(
            500,
            f"Internal error. Quote request_id {REQUEST_ID.get()} when reporting this.",
            "internal_error")
