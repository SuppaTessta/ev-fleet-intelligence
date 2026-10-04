# Multi-stage: the wheel build needs compilers, the runtime does not.
# Final image ships no build toolchain and runs as a non-root user.
FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt .
# --no-cache-dir keeps the layer down; tensorflow-cpu is already ~500 MB
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

# libgomp1 is required by LightGBM at runtime and is NOT in python:slim.
# Without it, `import lightgbm` fails with a bare OSError about libgomp.so.1 --
# an error that looks nothing like a missing system package.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app
# Only what serving needs. This image previously also carried dashboard/,
# train/ and evaluation/ -- the Streamlit UI (which has its own image), the
# training pipelines and the evaluation harness. Nothing under backend/ imports
# them, so they were pure attack surface and image weight in a container whose
# job is to answer HTTP.
COPY backend/ ./backend/
COPY data/processed/ ./data/processed/

# Non-root. The app only ever reads its artifacts, so it needs no write access
# to anything it ships with.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
# Hits /health (liveness, no disk I/O) rather than /ready: a container with a
# missing model artifact is still correctly *running*, and should report as
# healthy-but-degraded via /ready rather than being killed and restarted forever.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=4).status==200 else 1)"

WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
