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

TEMPERATURE="${TEMPERATURE:-0}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-vllm/results/raw/${RUN_ID}}"

mkdir -p "$RUN_DIR"

echo "RUN_ID=$RUN_ID"
echo "RUN_DIR=$RUN_DIR"
echo
echo "Running vLLM serving benchmark"
echo "MODEL=$MODEL"
echo "PORT=$PORT"
echo "NUM_PROMPTS=$NUM_PROMPTS"
echo "NUM_WARMUPS=$NUM_WARMUPS"
echo "REQUEST_RATE=$REQUEST_RATE"
echo "BURSTINESS=$BURSTINESS"
echo "MAX_CONCURRENCY=${MAX_CONCURRENCY:-unlimited}"
echo "INPUT_LEN=$INPUT_LEN"
echo "OUTPUT_LEN=$OUTPUT_LEN"
echo "TEMPERATURE=$TEMPERATURE"

CMD=(
  vllm bench serve
  --backend openai-chat
  --model "$MODEL"
  --host 127.0.0.1
  --port "$PORT"
  --endpoint /v1/chat/completions

  --dataset-name random
  --random-input-len "$INPUT_LEN"
  --random-output-len "$OUTPUT_LEN"
  --random-range-ratio 0.0

  --num-prompts "$NUM_PROMPTS"
  --num-warmups "$NUM_WARMUPS"

  --request-rate "$REQUEST_RATE"
  --burstiness "$BURSTINESS"

  --temperature "$TEMPERATURE"
  --ignore-eos

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
MODEL=$MODEL
PORT=$PORT
NUM_PROMPTS=$NUM_PROMPTS
NUM_WARMUPS=$NUM_WARMUPS
REQUEST_RATE=$REQUEST_RATE
BURSTINESS=$BURSTINESS
MAX_CONCURRENCY=${MAX_CONCURRENCY:-unlimited}
INPUT_LEN=$INPUT_LEN
OUTPUT_LEN=$OUTPUT_LEN
RANDOM_RANGE_RATIO=0.0
TEMPERATURE=$TEMPERATURE
IGNORE_EOS=true
EOF

{
  printf '%q ' "${CMD[@]}"
  printf '\n'
} > "$RUN_DIR/benchmark-command.txt"

echo
echo "Exact benchmark command:"
cat "$RUN_DIR/benchmark-command.txt"
echo

"${CMD[@]}" 2>&1 | tee "$RUN_DIR/benchmark.log"

echo
echo "Benchmark complete."
echo "Results:"
echo "$RUN_DIR/benchmark.json"