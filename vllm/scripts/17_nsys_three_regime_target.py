#!/usr/bin/env python3

import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

import torch


MODEL = os.environ.get(
    "MODEL",
    "Qwen/Qwen2.5-3B-Instruct",
)

MODEL_REVISION = os.environ["MODEL_REVISION"]

PORT = int(
    os.environ.get("PORT", "8000")
)

GPU_UTIL = os.environ.get(
    "GPU_UTIL",
    "0.50",
)

MAX_MODEL_LEN = os.environ.get(
    "MAX_MODEL_LEN",
    "8192",
)

INPUT_LEN = os.environ.get(
    "INPUT_LEN",
    "512",
)

OUTPUT_LEN = os.environ.get(
    "OUTPUT_LEN",
    "128",
)

SEED = os.environ.get(
    "SEED",
    "0",
)

WAVES_PER_POINT = int(
    os.environ.get(
        "WAVES_PER_POINT",
        "8",
    )
)

SERVER_READY_TIMEOUT_S = int(
    os.environ.get(
        "SERVER_READY_TIMEOUT_S",
        "420",
    )
)

RUN_DIR = Path(
    os.environ["RUN_DIR"]
)

RUN_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def wait_for_health():
    deadline = (
        time.monotonic()
        + SERVER_READY_TIMEOUT_S
    )

    url = (
        f"http://127.0.0.1:"
        f"{PORT}/health"
    )

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(
                url,
                timeout=1,
            ) as response:
                if response.status == 200:
                    return
        except Exception:
            pass

        time.sleep(1)

    raise RuntimeError(
        "vLLM readiness timeout"
    )


def run_benchmark(
    *,
    concurrency,
    num_prompts,
    point_dir,
    nvtx_name,
):
    point_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    env = os.environ.copy()

    env.update(
        {
            "RUN_ID":
                f"nsys_{nvtx_name}",

            "RUN_DIR":
                str(point_dir),

            "MODEL":
                MODEL,

            "PORT":
                str(PORT),

            "NUM_PROMPTS":
                str(num_prompts),

            "NUM_WARMUPS":
                "0",

            "REQUEST_RATE":
                "inf",

            "MAX_CONCURRENCY":
                str(concurrency),

            "INPUT_LEN":
                INPUT_LEN,

            "OUTPUT_LEN":
                OUTPUT_LEN,

            "SEED":
                SEED,
        }
    )

    print()
    print(
        f"Starting {nvtx_name}: "
        f"concurrency={concurrency}, "
        f"prompts={num_prompts}"
    )

    # This NVTX range exists on the same global
    # Nsight timeline as the EngineCore GPU work.
    # We will use its timestamps later to slice
    # the GPU trace.
    torch.cuda.nvtx.range_push(
        nvtx_name
    )

    try:
        subprocess.run(
            [
                "bash",
                "vllm/scripts/"
                "02_bench_random_baseline.sh",
            ],
            env=env,
            check=True,
        )
    finally:
        torch.cuda.nvtx.range_pop()

    print(
        f"Completed {nvtx_name}"
    )


server_log_path = (
    RUN_DIR / "server.log"
)

server_log = server_log_path.open(
    "w"
)

server_cmd = [
    "vllm",
    "serve",
    MODEL,

    "--host",
    "127.0.0.1",

    "--port",
    str(PORT),

    "--gpu-memory-utilization",
    GPU_UTIL,

    "--max-model-len",
    MAX_MODEL_LEN,

    "--generation-config",
    "vllm",

    "--revision",
    MODEL_REVISION,

    "--no-enable-prefix-caching",
]

print("Starting vLLM server...")

server = subprocess.Popen(
    server_cmd,
    stdout=server_log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)

try:
    wait_for_health()

    print("Server healthy.")

    # Unmeasured warmup.
    run_benchmark(
        concurrency=64,
        num_prompts=64,
        point_dir=RUN_DIR / "_warmup",
        nvtx_name="NSYS_WARMUP",
    )

    for concurrency in (
        64,
        128,
        256,
    ):
        num_prompts = (
            concurrency
            * WAVES_PER_POINT
        )

        run_benchmark(
            concurrency=concurrency,
            num_prompts=num_prompts,
            point_dir=(
                RUN_DIR
                / f"concurrency_{concurrency:03d}"
            ),
            nvtx_name=(
                f"NSYS_REGIME_C"
                f"{concurrency}"
            ),
        )

finally:
    print()
    print("Stopping vLLM server...")

    try:
        os.killpg(
            server.pid,
            signal.SIGTERM,
        )
    except ProcessLookupError:
        pass

    try:
        server.wait(
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(
                server.pid,
                signal.SIGKILL,
            )
        except ProcessLookupError:
            pass

        server.wait()

    server_log.close()

print("Three-regime target complete.")