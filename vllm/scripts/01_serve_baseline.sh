#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MODEL_REVISION="${MODEL_REVISION:-}"

# default = leave vLLM's own default untouched
# on      = explicitly enable prefix caching
# off     = explicitly disable prefix caching
PREFIX_CACHING="${PREFIX_CACHING:-default}"

RUN_ID="${RUN_ID:-001_vllm_baseline_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-vllm/results/raw/$RUN_ID}"

mkdir -p "$RUN_DIR"

case "$PREFIX_CACHING" in
  default|on|off)
    ;;
  *)
    echo "PREFIX_CACHING must be: default, on, or off"
    exit 1
    ;;
esac

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

case "$PREFIX_CACHING" in
  on)
    CMD+=(--enable-prefix-caching)
    ;;
  off)
    CMD+=(--no-enable-prefix-caching)
    ;;
esac

cat > "$RUN_DIR/server-config.txt" <<EOF
model=$MODEL
model_revision=$MODEL_REVISION
port=$PORT
gpu_memory_utilization=$GPU_UTIL
max_model_len=$MAX_MODEL_LEN
generation_config=vllm
prefix_caching=$PREFIX_CACHING
EOF

printf '%q ' "${CMD[@]}" > "$RUN_DIR/server-command.txt"
printf '\n' >> "$RUN_DIR/server-command.txt"

echo "========================================"
echo "Starting vLLM server"
echo "========================================"
echo "MODEL=$MODEL"
echo "MODEL_REVISION=$MODEL_REVISION"
echo "PORT=$PORT"
echo "GPU_UTIL=$GPU_UTIL"
echo "MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "PREFIX_CACHING=$PREFIX_CACHING"
echo "RUN_DIR=$RUN_DIR"
echo

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/server.log"