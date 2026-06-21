#!/usr/bin/env python3
"""
Pre-download model weights for one or more backends, so the first
real generation request doesn't stall for 10-40 minutes downloading
weights mid-request.

Usage:
    python3 scripts/download_models.py --model ltx-video
    python3 scripts/download_models.py --model wan2.1 --model svd
    python3 scripts/download_models.py --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.core.config import settings  # noqa: E402

REPOS = {
    "ltx-video": ["Lightricks/LTX-Video"],
    "wan2.1": [
        "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    ],
    "hunyuan-video": ["tencent/HunyuanVideo"],
    "svd": [
        "stabilityai/stable-video-diffusion-img2vid-xt",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", choices=list(REPOS), default=[])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    targets = list(REPOS) if args.all else args.model
    if not targets:
        parser.error("Specify --model <name> (one or more) or --all")

    from huggingface_hub import snapshot_download

    for model in targets:
        for repo in REPOS[model]:
            print(f"==> Downloading {repo} (for backend '{model}') ...")
            snapshot_download(
                repo_id=repo,
                cache_dir=str(settings.models_cache_dir / "hf"),
                token=settings.hf_token,
            )
            print(f"    done: {repo}")

    print("\nAll requested model weights are cached locally.")


if __name__ == "__main__":
    main()
