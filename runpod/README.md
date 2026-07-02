# RunPod Execution Guide

This repository is edited locally but executed on a cloud GPU environment.

The intended workflow is:

1. Edit scripts and notes locally.
2. Push changes to GitHub.
3. Start a RunPod GPU pod.
4. Clone this repository inside the pod.
5. Run vLLM serving and benchmark scripts.
6. Save raw benchmark outputs under `vllm/results/raw/`.
7. Summarize important results in notes and README files.

## Why RunPod?

Local machines often differ in GPU, driver, CUDA, Python, storage, and thermal behavior.

For inference performance work, the execution environment should be explicit and repeatable:

- GPU model
- VRAM
- driver version
- CUDA version
- Python version
- vLLM version
- model name
- benchmark command
- workload shape

## First target environment

The first benchmark target is a single-GPU RunPod pod.

Recommended initial model:

- `Qwen/Qwen2.5-7B-Instruct` on a 40GB+ GPU
- `Qwen/Qwen2.5-3B-Instruct` on a smaller GPU

## Reproducibility checklist

Record the following for every run:

```bash
nvidia-smi
python --version
vllm --version
pip freeze | grep -E "vllm|torch|transformers"