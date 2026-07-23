#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "Starting vLLM server"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "GPU_UTIL=$GPU_UTIL"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_MODEL_LEN"