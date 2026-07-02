#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${PORT:-8000}"
NUM_PROMPTS="${NUM_PROMPTS:-128}"
REQUEST_RATE="${REQUEST_RATE:-4}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"

mkdir -p vllm/results/raw

OUT="vllm/results/raw/random_baseline_np${NUM_PROMPTS}_rr${REQUEST_RATE}_in${INPUT_LEN}_out${OUTPUT_LEN}.json"

echo "Running vLLM benchmark"
echo "MODEL=$MODEL"
echo "NUM_PROMPTS=$NUM_PROMPTS"
echo "REQUEST_RATE=$REQUEST_RATE"
echo "INPUT_LEN=$INPUT_LEN"
echo "OUTPUT_LEN=$OUTPUT_LEN"
echo "OUT=$OUT"

vllm bench serve \
  --backend openai-chat \
  --model "$MODEL" \
  --base-url "http://localhost:${PORT}" \
  --dataset-name random \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --num-prompts "$NUM_PROMPTS" \
  --request-rate "$REQUEST_RATE" \
  --save-result \
  --result-filename "$OUT"