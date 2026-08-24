# RunPod Execution Guide

This directory documents how to run the LLM Inference Performance Lab on
reproducible remote GPU infrastructure.

The current workflow is intentionally manual.

The goal is to understand every stage of the experiment before automating
provisioning, container builds, benchmark execution, result collection,
and Pod destruction.

## Current Architecture

```text
Local machine
    |
    | git push
    v
GitHub repository
    |
    | git clone / checkout exact commit
    v
RunPod GPU Pod
    |
    | setup_runpod.sh
    v
Controlled Python environment
    |
    | 03_run_baseline.sh
    v
Environment capture
    |
    v
vLLM server
    |
    v
Health check
    |
    v
Benchmark
    |
    v
Raw experiment artifacts
```

The local laptop is the development and control machine.

The RunPod GPU is the benchmark machine.

Performance numbers from the laptop are not used as benchmark evidence.

## First GPU Target

For the first baseline:

```text
Cloud: RunPod Secure Cloud
GPU count: 1
GPU class: NVIDIA A100 80 GB
Template: Official RunPod PyTorch template
Engine: vLLM 0.27.1
Model: Qwen/Qwen2.5-3B-Instruct
```

The first run is a smoke test and reproducibility test, not a performance
claim.

For later formal benchmarks, the exact GPU SKU must be recorded and kept
constant when comparing configurations.

RunPod Secure Cloud is preferred for formal performance work because the
infrastructure is operated in datacenter environments with higher
reliability than Community Cloud.

## 1. Prepare the Repository Locally

Before creating the GPU Pod:

```bash
git status
git log -1 --oneline
git push origin main
```

For a formal benchmark, the working tree should be clean.

Record the commit that will be executed:

```bash
git rev-parse HEAD
```

## 2. Create the RunPod Pod

In the RunPod console:

```text
Pod type: Secure Cloud
GPU: 1 x A100 80 GB
Template: Official RunPod PyTorch
```

Do not expose the vLLM API publicly for this baseline.

The benchmark client and vLLM server run on the same Pod and communicate
through localhost.

After the Pod starts, connect through SSH or the RunPod web terminal.

## 3. Clone the Exact Repository State

On the Pod:

```bash
cd /workspace

git clone https://github.com/cengizasmazoglu/llm-inference-perf-lab.git

cd llm-inference-perf-lab
```

Confirm the revision:

```bash
git rev-parse HEAD
```

It should match the commit intended for the experiment.

For a formal experiment, an exact commit can be checked out explicitly:

```bash
git checkout <COMMIT_SHA>
```

## 4. Prepare the Controlled Environment

Run:

```bash
bash runpod/setup_runpod.sh
```

This creates:

```text
.venv-runpod/
```

and installs the pinned lab dependencies into that environment.

Activate it:

```bash
source .venv-runpod/bin/activate
```

Verify:

```bash
which python
python --version
vllm --version
nvidia-smi
```

## 5. Run Experiment 001

From the repository root:

```bash
bash vllm/scripts/03_run_baseline.sh
```

The runner:

```text
creates one RUN_ID
        |
        v
captures environment information
        |
        v
starts vLLM
        |
        v
waits for /health
        |
        v
runs the controlled benchmark
        |
        v
stores artifacts under one RUN_DIR
        |
        v
stops the vLLM server
```

A run directory will look approximately like:

```text
vllm/results/raw/
└── 001_vllm_baseline_<UTC_TIMESTAMP>/
    ├── system.txt
    ├── nvidia-smi-q.txt
    ├── vllm-collect-env.txt
    ├── pip-freeze.txt
    ├── server-config.txt
    ├── server-command.txt
    ├── server.log
    ├── server.pid
    ├── benchmark-config.txt
    ├── benchmark-command.txt
    ├── benchmark.log
    └── benchmark.json
```

Additional telemetry and manifests will be added later.

## 6. Important Experimental Rule

The first successful execution is not evidence that one engine or
configuration is faster than another.

A performance claim requires controlled comparisons.

Examples of variables that must eventually be controlled or explicitly
varied include:

```text
GPU
engine version
container/environment
model revision
input length
output length
request rate
arrival distribution
concurrency
warmup
random seed
engine configuration
number of repetitions
```

Change one experimental dimension deliberately and record the change.

## Planned Reproducibility Upgrades

The current manual workflow is temporary.

After the underlying workflow is understood, the lab will evolve toward:

```text
Git commit
    |
    v
GitHub Actions
    |
    v
build benchmark container
    |
    v
GitHub Container Registry (GHCR)
    |
    v
immutable image digest
    |
    v
RunPod
```

Experiment tracking will also move toward an MLOps layer such as MLflow,
while retaining raw artifacts and exact configuration files.

Later infrastructure automation will perform:

```text
query GPU availability
        |
        v
create RunPod
        |
        v
checkout exact commit / pull exact image
        |
        v
run experiment
        |
        v
collect and persist results
        |
        v
destroy RunPod
```

The purpose of automation is to remove repetitive human operations after
those operations are understood, not to hide the experimental procedure.

## Future Performance Tooling

Later experiments will add:

```text
GPU utilization and VRAM time-series telemetry
Nsight Systems (nsys)
Nsight Compute (ncu)
controlled concurrency sweeps
request-rate sweeps
Gamma / Poisson / bursty arrival processes
variable prompt and output-length distributions
vLLM vs SGLang controlled comparisons
scheduler and KV-cache experiments
agent/session-style workloads
```

These are not required for Experiment 001.