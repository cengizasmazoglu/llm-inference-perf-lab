#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8000}"

NUM_PROMPTS="${NUM_PROMPTS:-128}"
NUM_WARMUPS="${NUM_WARMUPS:-8}"

REQUEST_RATE="${REQUEST_RATE:-4}"
BURSTINESS="${BURSTINESS:-1.0}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-}"

INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

SEED="${SEED:-0}"

RUN_ID="${RUN_ID:-001_vllm_baseline_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-vllm/results/raw/$RUN_ID}"

mkdir -p "$RUN_DIR"

CMD=(
  vllm bench serve
  --backend openai-chat
  --model "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --endpoint /v1/chat/completions
  --dataset-name random
  --num-prompts "$NUM_PROMPTS"
  --num-warmups "$NUM_WARMUPS"
  --request-rate "$REQUEST_RATE"
  --burstiness "$BURSTINESS"
  --random-input-len "$INPUT_LEN"
  --random-output-len "$OUTPUT_LEN"
  --random-range-ratio 0.0
  --seed "$SEED"
  --ignore-eos
  --temperature 0
  --percentile-metrics ttft,tpot,itl,e2el
  --metric-percentiles 50,95,99
  --save-result
  --save-detailed
  --result-dir "$RUN_DIR"
  --result-filename benchmark.json
  --metadata "run_id=$RUN_ID"
)

if [[ -n "$MAX_CONCURRENCY" ]]; then
  CMD+=(--max-concurrency "$MAX_CONCURRENCY")
fi

cat > "$RUN_DIR/benchmark-config.txt" <<EOF
model=$MODEL
num_prompts=$NUM_PROMPTS
num_warmups=$NUM_WARMUPS
request_rate=$REQUEST_RATE
burstiness=$BURSTINESS
max_concurrency=$MAX_CONCURRENCY
input_len=$INPUT_LEN
output_len=$OUTPUT_LEN
seed=$SEED
temperature=0
ignore_eos=true
EOF

printf '%q ' "${CMD[@]}" > "$RUN_DIR/benchmark-command.txt"
printf '\n' >> "$RUN_DIR/benchmark-command.txt"

echo "========================================"
echo "Serving benchmark"
echo "========================================"
echo "RUN_ID=$RUN_ID"
echo "REQUEST_RATE=$REQUEST_RATE"
echo "MAX_CONCURRENCY=${MAX_CONCURRENCY:-unlimited}"
echo "NUM_PROMPTS=$NUM_PROMPTS"
echo "SEED=$SEED"
echo

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/benchmark.log"

echo
echo "Benchmark complete."
echo "Results:"
echo "$RUN_DIR/benchmark.json"