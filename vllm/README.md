# vLLM Experiments

This module measures vLLM serving behavior under controlled workloads.

The first experiments focus on:

- TTFT: time to first token
- TPOT: time per output token / inter-token latency
- request throughput
- output tokens per second
- p95 latency
- GPU memory usage
- sensitivity to input length, output length, and request rate

## Experiment 001: Baseline

Start the vLLM server:

```bash
MODEL=Qwen/Qwen2.5-3B-Instruct \
GPU_UTIL=0.50 \
MAX_MODEL_LEN=8192 \
bash vllm/scripts/01_serve_baseline.sh