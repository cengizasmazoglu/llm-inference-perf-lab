# LLM Inference Performance Lab

Reproducible experiments for measuring and tuning LLM inference serving systems.

## Article #1 — vLLM Concurrency Knee on A100

**Beyond Peak Throughput: Finding and Explaining vLLM’s Concurrency Knee on an A100**

A controlled saturation study of vLLM 0.27.1 with Qwen2.5-3B-Instruct on an NVIDIA A100-SXM4-80GB.

Across three balanced repeats, output throughput peaked at concurrency 128 and declined at 256, while TTFT and TPOT deteriorated sharply. A fresh-host clean-room reproduction preserved the same regime ordering.

Nsight Systems analysis localizes the additional post-knee GPU cost to a sustained mixed/prefill-containing FlashAttention path, while inspection of the vLLM V1 scheduler provides the systems-level connection to continuous batching under a fixed token budget.

**Frozen evidence release:** [article1-v1.0](https://github.com/cengizasmazoglu/llm-inference-perf-lab/releases/tag/article1-v1.0)

**Publication figures and evidence:** [`vllm/results/article1/`](vllm/results/article1/)

**Article:** forthcoming on ProduckAI

This repository focuses on production-relevant inference metrics:

- TTFT: time to first token
- TPOT / inter-token latency
- p95 / p99 latency
- request throughput
- output tokens per second
- GPU utilization
- KV cache pressure
- memory usage
- cost-per-token proxy

The first module focuses on vLLM. Later modules will add SGLang and TensorRT-LLM.

## Why this exists

LLM inference performance is not one problem.

Interactive workloads care about TTFT and tail latency.
Batch/offline workloads care about throughput.
Long-context workloads create KV-cache pressure.
Mixed prompt-length workloads stress the scheduler.
Naive benchmarks hide these tradeoffs.

This lab measures those tradeoffs explicitly.

## Current status

- [x] Controlled vLLM saturation benchmark on A100
- [x] Three-repeat concurrency-knee validation
- [x] Fresh-host clean-room reproduction
- [x] Nsight Systems mechanism analysis
- [x] Frozen Article #1 evidence release
- [ ] vLLM vs SGLang comparison
- [ ] TensorRT-LLM investigation

## Author

Cengiz Asmazoglu  
PhD Researcher · LLM Inference Performance Engineer
