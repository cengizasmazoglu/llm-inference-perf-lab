# LLM Inference Performance Lab

Reproducible experiments for measuring and tuning LLM inference serving systems.

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

- [ ] vLLM baseline benchmark
- [ ] vLLM batching parameter sweep
- [ ] vLLM chunked prefill experiment
- [ ] vLLM prefix caching experiment
- [ ] vLLM metrics dashboard
- [ ] SGLang comparison
- [ ] TensorRT-LLM comparison

## Author

Cengiz Asmazoğlu  
LLM Inference Performance Consultant  
PhD researcher in LLM inference serving optimization
