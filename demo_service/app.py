"""FastAPI app for CoolPrompt Interface Demo."""

from __future__ import annotations

import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import FastAPI, HTTPException
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

app = FastAPI(title=settings.app_name, version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _store(job: JobStatus) -> None:
    with jobs_lock:
        jobs[job.job_id] = job


def _patch_job(job_id: str, **updates: Any) -> None:
    with jobs_lock:
        job = jobs[job_id]
        data = job.model_dump()
        data.update(updates)
        data["updated_at"] = time.time()
        jobs[job_id] = JobStatus(**data)


def _run_job(job_id: str, payload: JobCreateRequest) -> None:
    _patch_job(job_id, status="running")
    try:
        if payload.mode == "single":
            assert payload.request is not None
            result = run_single_optimization(payload.request, settings)
        else:
            assert payload.compare is not None
            result = run_comparison(payload.compare, settings)
        _patch_job(job_id, status="completed", result=result)
    except Exception as exc:  # noqa: BLE001 - surfaced to UI as job error
        _patch_job(job_id, status="failed", error=str(exc))


@app.get("/")
def index() -> FileResponse:
    """Serve the one-page demo interface."""

    return FileResponse(STATIC_DIR / "index.html")


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
    )
    _store(job)
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
