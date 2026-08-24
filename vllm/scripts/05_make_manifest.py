#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def command(*args):
    try:
        return subprocess.check_output(
            args,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def parse_key_value_file(path):
    result = {}

    if not path.exists():
        return result

    for line in path.read_text().splitlines():
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        result[key.strip()] = value.strip()

    return result


def scalar_summary(path):
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    return {
        key: value
        for key, value in data.items()
        if value is None or isinstance(value, (str, int, float, bool))
    }


if len(sys.argv) != 2:
    print("Usage: 05_make_manifest.py <RUN_DIR>", file=sys.stderr)
    sys.exit(1)


run_dir = Path(sys.argv[1])

if not run_dir.exists():
    raise SystemExit(f"Run directory does not exist: {run_dir}")


server_config = parse_key_value_file(
    run_dir / "server-config.txt"
)

benchmark_config = parse_key_value_file(
    run_dir / "benchmark-config.txt"
)


# ---------------------------------------------------------
# Git identity
# ---------------------------------------------------------

git_commit = command("git", "rev-parse", "HEAD")
git_branch = command("git", "branch", "--show-current")
git_status = command("git", "status", "--porcelain")

git_dirty = bool(git_status)


# ---------------------------------------------------------
# Python / vLLM / PyTorch / CUDA
# ---------------------------------------------------------

software = {}

try:
    import torch
    import vllm

    software = {
        "python": sys.version.split()[0],
        "vllm": vllm.__version__,
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
    }

    gpus = []

    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)

        gpus.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_gib": round(
                    props.total_memory / (1024 ** 3),
                    2,
                ),
                "compute_capability": (
                    f"{props.major}.{props.minor}"
                ),
            }
        )

except Exception as exc:
    software = {
        "error": str(exc),
    }
    gpus = []


# ---------------------------------------------------------
# Resolve the exact Hugging Face model commit
# ---------------------------------------------------------

model_name = server_config.get("MODEL")
requested_revision = server_config.get(
    "MODEL_REVISION",
    "unpinned",
)

resolved_model_sha = None
model_resolution_error = None

if model_name:
    try:
        from huggingface_hub import HfApi

        revision = (
            "main"
            if requested_revision in ("", "unpinned")
            else requested_revision
        )

        info = HfApi().model_info(
            model_name,
            revision=revision,
        )

        resolved_model_sha = info.sha

    except Exception as exc:
        model_resolution_error = str(exc)


# ---------------------------------------------------------
# RunPod metadata
# ---------------------------------------------------------

runpod_metadata = {
    key: value
    for key, value in sorted(os.environ.items())
    if key.startswith("RUNPOD_")
}


# ---------------------------------------------------------
# Container information
#
# Exact immutable digest will become authoritative after
# we move to our own GHCR-built benchmark containers.
# ---------------------------------------------------------

container = {
    "image": os.environ.get("RUNPOD_CONTAINER_IMAGE"),
    "digest": os.environ.get("RUNPOD_CONTAINER_IMAGE_DIGEST"),
    "digest_status": (
        "captured"
        if os.environ.get("RUNPOD_CONTAINER_IMAGE_DIGEST")
        else "not_available_from_current_manual_environment"
    ),
}


# ---------------------------------------------------------
# Benchmark summary
# ---------------------------------------------------------

benchmark_summary = scalar_summary(
    run_dir / "benchmark.json"
)


# ---------------------------------------------------------
# Artifacts
# ---------------------------------------------------------

artifacts = sorted(
    path.name
    for path in run_dir.iterdir()
    if path.is_file()
)


manifest = {
    "schema_version": 1,

    "run": {
        "run_id": os.environ.get(
            "RUN_ID",
            run_dir.name,
        ),
        "generated_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    },

    "source": {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
    },

    "software": software,

    "hardware": {
        "gpus": gpus,
    },

    "model": {
        "name": model_name,
        "requested_revision": requested_revision,
        "resolved_huggingface_sha": resolved_model_sha,
        "resolution_error": model_resolution_error,
    },

    "server": server_config,

    "workload": benchmark_config,

    "benchmark_summary": benchmark_summary,

    "runpod": runpod_metadata,

    "container": container,

    "artifacts": artifacts,
}


out = run_dir / "manifest.json"

out.write_text(
    json.dumps(
        manifest,
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

print(f"Manifest written: {out}")