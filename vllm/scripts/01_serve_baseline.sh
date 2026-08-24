#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MODEL_REVISION="${MODEL_REVISION:-}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-vllm/results/raw/${RUN_ID}}"

mkdir -p "$RUN_DIR"

echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo
echo "Starting vLLM server"
echo "MODEL=$MODEL"
echo "MODEL_REVISION=${MODEL_REVISION:-unpinned}"
echo "PORT=$PORT"
echo "GPU_UTIL=$GPU_UTIL"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"

CMD=(
  vllm serve "$MODEL"
  --host 0.0.0.0
  --port "$PORT"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_MODEL_LEN"
  --generation-config vllm
)

if [[ -n "$MODEL_REVISION" ]]; then
  CMD+=(--revision "$MODEL_REVISION")
fi

cat > "$RUN_DIR/server-config.txt" <<EOF
MODEL=$MODEL
MODEL_REVISION=${MODEL_REVISION:-unpinned}
PORT=$PORT
GPU_UTIL=$GPU_UTIL
MAX_MODEL_LEN=$MAX_MODEL_LEN
GENERATION_CONFIG=vllm
EOF

{
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$RUN_DIR/server-command.txt"

echo
echo "Exact server command:"
cat "$RUN_DIR/server-command.txt"

echo
echo "Server log:"
echo "$RUN_DIR/server.log"
echo

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/server.log"