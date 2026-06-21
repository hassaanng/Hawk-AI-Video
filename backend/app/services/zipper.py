"""
Builds a single ZIP archive containing every successfully generated
video in a batch, for the one-click "Download All" button.
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path

from app.core.config import settings
from app.services.job_store import JobStatus, job_store

logger = logging.getLogger(__name__)


async def build_batch_zip(batch_id: str) -> Path | None:
    jobs = await job_store.get_jobs_for_batch(batch_id)
    done_jobs = [j for j in jobs if j["status"] == JobStatus.DONE.value and j["output_path"]]

    if not done_jobs:
        logger.warning("No completed jobs for batch %s — skipping zip build.", batch_id)
        return None

    zip_path = settings.outputs_dir / f"batch_{batch_id}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for job in done_jobs:
            src = Path(job["output_path"])
            if not src.exists():
                logger.warning("Output file missing for job %s: %s", job["id"], src)
                continue
            arcname = f"{job['idx_in_batch']:03d}_{job['id'][:8]}.mp4"
            zf.write(src, arcname=arcname)

    logger.info("Built batch zip for %s with %d video(s) at %s", batch_id, len(done_jobs), zip_path)
    return zip_path
