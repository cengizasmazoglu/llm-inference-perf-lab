# 001 — vLLM Baseline

## Goal

Establish a reproducible vLLM serving baseline before tuning any parameter.

## Model

TBD

## Hardware

TBD

## Workload

- Dataset: random
- Input length: 512 tokens
- Output length: 128 tokens
- Number of prompts: 128
- Request rate: 4 requests/sec

## Metrics to collect

- TTFT
- TPOT / inter-token latency
- request throughput
- output tokens/sec
- p95 latency
- GPU memory usage

## Notes

The goal of this first experiment is not to optimize.

The goal is to verify that the benchmark is reproducible and that we can collect the right metrics.