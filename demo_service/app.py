"""FastAPI app for CoolPrompt Interface Demo."""

from __future__ import annotations

import time
import uuid
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .methods import METHOD_BY_ID, public_methods
from .runner import run_comparison, run_single_optimization
from .schemas import JobCreateRequest, JobStatus
from .settings import get_settings


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

settings = get_settings()
executor = ThreadPoolExecutor(max_workers=settings.max_workers)
jobs: dict[str, JobStatus] = {}
jobs_lock = Lock()
logger = logging.getLogger("demo_service")
logger.setLevel(logging.INFO)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _model_options() -> list[dict[str, str]]:
    base_url = (settings.openai_base_url or "").lower()
    if "openrouter" in base_url:
        return [
            {"value": "gpt-4o-mini", "label": "gpt-4o-mini"},
            {"value": "gpt-4o", "label": "gpt-4o"},
            {"value": "google/gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
            {"value": "anthropic/claude-3.5-haiku", "label": "Claude 3.5 Haiku"},
        ]
    return [
        {"value": "gpt-4o-mini", "label": "gpt-4o-mini"},
        {"value": "gpt-4o", "label": "gpt-4o"},
        {"value": "gpt-4.1-mini", "label": "gpt-4.1-mini"},
        {"value": "gpt-4.1-nano", "label": "gpt-4.1-nano"},
    ]


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store, max-age=0"
    return response


def _store(job: JobStatus) -> None:
    with jobs_lock:
        jobs[job.job_id] = job


def _patch_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        job = jobs[job_id]
        if job.status in {"completed", "failed"}:
            return
        data = job.model_dump()
        data.update(updates)
        data["updated_at"] = time.time()
        jobs[job_id] = JobStatus(**data)


def _run_job(job_id: str, payload: JobCreateRequest) -> None:
    def progress(stage: str, percent: int, message: str) -> None:
        logger.info("job=%s stage=%s percent=%s message=%s", job_id, stage, percent, message)
        _patch_job(
            job_id,
            status="running",
            progress_stage=stage,
            progress_percent=percent,
            progress_message=message,
        )

    progress("preparing", 10, "Готовим запуск")
    try:
        logger.info("job=%s mode=%s started", job_id, payload.mode)
        if payload.mode == "single":
            assert payload.request is not None
            result = run_single_optimization(payload.request, settings, progress_callback=progress)
        else:
            assert payload.compare is not None
            result = run_comparison(payload.compare, settings, progress_callback=progress)
        _patch_job(
            job_id,
            status="completed",
            progress_stage="completed",
            progress_percent=100,
            progress_message="Оптимизация завершена",
            result=result,
        )
        logger.info("job=%s completed", job_id)
    except Exception as exc:  # noqa: BLE001 - surfaced to UI as job error
        logger.exception("job=%s failed", job_id)
        _patch_job(
            job_id,
            status="failed",
            progress_stage="failed",
            progress_percent=100,
            progress_message="Оптимизация завершилась с ошибкой",
            error=str(exc),
        )


@app.get("/")
def index() -> FileResponse:
    """Serve the one-page demo interface."""

    return FileResponse(STATIC_DIR / "index.html")


@app.head("/")
def head_index() -> Response:
    """Render probes the root URL with HEAD before declaring the service live."""

    return Response(status_code=200)


@app.get("/api/health")
def health() -> dict[str, str]:
    """Railway healthcheck endpoint."""

    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    """Expose non-secret frontend configuration."""

    return {
        "appName": settings.app_name,
        "defaultModel": settings.model_name,
        "hasOpenAIKey": settings.has_openai_key,
        "allowMock": settings.allow_mock,
        "forceMock": settings.force_mock,
        "maxCompareMethods": settings.max_compare_methods,
        "modelOptions": _model_options(),
    }


@app.get("/api/methods")
def methods() -> list[dict[str, Any]]:
    """Return method catalog for the UI."""

    return public_methods()


@app.post("/api/jobs", response_model=JobStatus)
def create_job(payload: JobCreateRequest) -> JobStatus:
    """Create a background optimization or comparison job."""

    if payload.mode == "single" and payload.request is not None:
        if payload.request.method not in METHOD_BY_ID:
            raise HTTPException(status_code=400, detail=f"Unknown method: {payload.request.method}")
    if payload.mode == "compare" and payload.compare is not None:
        unknown = [method for method in payload.compare.methods if method not in METHOD_BY_ID]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown method(s): {', '.join(unknown)}")

    job = JobStatus(
        job_id=uuid.uuid4().hex,
        mode=payload.mode,
        status="queued",
        created_at=time.time(),
        updated_at=time.time(),
        progress_stage="queued",
        progress_percent=5,
        progress_message="Задача ожидает запуска",
    )
    _store(job)
    if payload.mode == "single" and payload.request is not None:
        logger.info(
            "job=%s queued mode=single method=%s model=%s mock=%s dataset_size=%s",
            job.job_id,
            payload.request.method,
            payload.request.model_name or settings.model_name,
            payload.request.mock or settings.force_mock,
            len(payload.request.dataset or []),
        )
    else:
        logger.info("job=%s queued mode=%s", job.job_id, payload.mode)
    future: Future = executor.submit(_run_job, job.job_id, payload)
    future.add_done_callback(lambda _: None)
    return job


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str) -> JobStatus:
    """Read the current job state."""

    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
