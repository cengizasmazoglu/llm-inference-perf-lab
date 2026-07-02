#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.90}"

echo "Starting vLLM server"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "GPU_UTIL=$GPU_UTIL"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL"