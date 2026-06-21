"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

The worker pool (which actually runs video generation in subprocesses)
is started/stopped via FastAPI's lifespan context, so `uvicorn --reload`
during development cleanly tears down GPU-holding processes on restart
instead of leaking them.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.services.job_store import job_store
from app.workers.batch_worker import worker_pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await job_store.init()
    requeued = await job_store.requeue_stale_running_jobs()
    if requeued:
        logger.warning(
            "Requeued %d job(s) that were left RUNNING by a previous, no-longer-running "
            "backend process. See docs/debugging.md if this happens unexpectedly often.",
            requeued,
        )
    await worker_pool.start()
    logger.info("AI Video Studio backend started (mode=%s).", settings.deployment_mode)
    yield
    await worker_pool.stop()
    logger.info("AI Video Studio backend shut down.")


app = FastAPI(title="AI Video Studio", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
async def root():
    return {"name": "AI Video Studio", "status": "ok", "mode": settings.deployment_mode}


@app.get("/health")
async def health():
    return {"status": "ok"}
