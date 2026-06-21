"""
Persistent job store (SQLite via aiosqlite).

Jobs survive process restarts — important for long batch runs where a
50-prompt job might take hours; nobody wants that state living only in
an in-memory dict that vanishes on a server reload. SQLite is the
right tool here: single-writer, embedded, zero ops overhead, and the
write volume (job state transitions, not video frames) is trivially
within its comfort zone.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import aiosqlite

from app.core.config import settings


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELED = "canceled"


class BatchStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    PARTIAL = "partial"  # some jobs succeeded, some failed
    CANCELED = "canceled"


SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    total INTEGER NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    zip_path TEXT
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL,
    idx_in_batch INTEGER NOT NULL,
    prompt TEXT NOT NULL,
    params_json TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    status_message TEXT,
    output_path TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    FOREIGN KEY (batch_id) REFERENCES batches(id)
);

CREATE INDEX IF NOT EXISTS idx_jobs_batch ON jobs(batch_id);
"""


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._initialized = False

    async def init(self) -> None:
        if self._initialized:
            return
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA)
            await db.commit()
        self._initialized = True

    # ---- batches -----------------------------------------------------

    async def create_batch(self, prompts: list[dict[str, Any]]) -> str:
        batch_id = str(uuid.uuid4())
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO batches (id, status, total, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (batch_id, BatchStatus.QUEUED.value, len(prompts), now, now),
            )
            for i, p in enumerate(prompts):
                job_id = str(uuid.uuid4())
                await db.execute(
                    """INSERT INTO jobs
                       (id, batch_id, idx_in_batch, prompt, params_json, status, progress, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)""",
                    (job_id, batch_id, i, p["prompt"], json.dumps(p), JobStatus.QUEUED.value, now, now),
                )
            await db.commit()
        return batch_id

    async def get_batch(self, batch_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM batches WHERE id = ?", (batch_id,))
            row = await cur.fetchone()
            return dict(row) if row else None

    async def list_batches(self, limit: int = 50) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM batches ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def update_batch_status(self, batch_id: str, status: BatchStatus, zip_path: str | None = None) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            if zip_path:
                await db.execute(
                    "UPDATE batches SET status = ?, updated_at = ?, zip_path = ? WHERE id = ?",
                    (status.value, time.time(), zip_path, batch_id),
                )
            else:
                await db.execute(
                    "UPDATE batches SET status = ?, updated_at = ? WHERE id = ?",
                    (status.value, time.time(), batch_id),
                )
            await db.commit()

    # ---- jobs ----------------------------------------------------------

    async def get_jobs_for_batch(self, batch_id: str) -> list[dict[str, Any]]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM jobs WHERE batch_id = ? ORDER BY idx_in_batch ASC", (batch_id,)
            )
            rows = await cur.fetchall()
            return [dict(r) for r in rows]

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
            row = await cur.fetchone()
            return dict(row) if row else None

    async def claim_next_queued_job(self, batch_id: str | None = None) -> dict[str, Any] | None:
        """Atomically claim the next QUEUED job (optionally scoped to one
        batch) by flipping it to RUNNING, so two worker loops can never
        pick up the same job."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if batch_id:
                cur = await db.execute(
                    "SELECT * FROM jobs WHERE status = ? AND batch_id = ? ORDER BY idx_in_batch ASC LIMIT 1",
                    (JobStatus.QUEUED.value, batch_id),
                )
            else:
                cur = await db.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at ASC LIMIT 1",
                    (JobStatus.QUEUED.value,),
                )
            row = await cur.fetchone()
            if not row:
                return None
            job = dict(row)
            await db.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (JobStatus.RUNNING.value, time.time(), job["id"], JobStatus.QUEUED.value),
            )
            await db.commit()
            # Re-check we actually won the race (rowcount-style check via re-fetch).
            cur2 = await db.execute("SELECT * FROM jobs WHERE id = ?", (job["id"],))
            confirmed = await cur2.fetchone()
            confirmed = dict(confirmed)
            if confirmed["status"] != JobStatus.RUNNING.value:
                return None
            return confirmed

    async def update_job_progress(self, job_id: str, progress: float, message: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE jobs SET progress = ?, status_message = ?, updated_at = ? WHERE id = ?",
                (progress, message, time.time(), job_id),
            )
            await db.commit()

    async def complete_job(self, job_id: str, output_path: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE jobs SET status = ?, progress = 1.0, output_path = ?, updated_at = ? WHERE id = ?",
                (JobStatus.DONE.value, output_path, time.time(), job_id),
            )
            await db.commit()

    async def fail_job(self, job_id: str, error: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (JobStatus.FAILED.value, error, time.time(), job_id),
            )
            await db.commit()

    async def requeue_stale_running_jobs(self) -> int:
        """
        On backend startup, any job still marked RUNNING necessarily
        belongs to a worker subprocess from a *previous* process
        lifetime — the in-memory WorkerPool that would have completed
        or failed it no longer exists. Without this, such jobs would
        sit RUNNING forever (see docs/debugging.md), permanently
        blocking their batch's ZIP finalization. Called once at
        startup in app/main.py's lifespan.
        """
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute(
                "SELECT id FROM jobs WHERE status = ?", (JobStatus.RUNNING.value,)
            )
            stale = [row[0] for row in await cur.fetchall()]
            if stale:
                await db.execute(
                    "UPDATE jobs SET status = ?, progress = 0, status_message = ?, updated_at = ? WHERE status = ?",
                    (JobStatus.QUEUED.value, "Requeued after server restart", time.time(), JobStatus.RUNNING.value),
                )
                await db.commit()
        return len(stale)


job_store = JobStore(settings.db_path)
