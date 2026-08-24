#!/usr/bin/env bash
set -euo pipefail

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${RUN_DIR:-vllm/results/raw/${RUN_ID}}"

mkdir -p "$RUN_DIR"

echo "RUN_ID: $RUN_ID"
echo "Run directory: $RUN_DIR"

{
  echo "=== BASIC SYSTEM ==="
  echo "UTC timestamp: $(date -u --iso-8601=seconds)"
  echo "Hostname: $(hostname)"
  echo

  echo "=== OS / KERNEL ==="
  uname -a
  echo

  echo "=== CPU ==="
  lscpu
  echo

  echo "=== MEMORY ==="
  free -h
  echo

  echo "=== PYTHON ==="
  python --version
  echo

  echo "=== GIT ==="
  echo "Commit:"
  git rev-parse HEAD
  echo
  echo "Branch:"
  git branch --show-current
  echo
  echo "Working tree:"
  git status --short
  echo

  echo "=== GPU SUMMARY ==="
  nvidia-smi -L || true
  echo

  echo "=== NVIDIA-SMI ==="
  nvidia-smi || true
  echo

  echo "=== PYTORCH / CUDA ==="
  python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime used by PyTorch:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"GPU {i}: {p.name}")
    print(f"GPU {i} total memory: {p.total_memory / 1024**3:.2f} GiB")
PY

  echo
  echo "=== vLLM ==="
  python - <<'PY'
import vllm
print(vllm.__version__)
PY

  echo
  echo "=== RUNPOD METADATA ==="
  env | sort | grep '^RUNPOD_' || true

} | tee "$RUN_DIR/system.txt"

echo "Collecting detailed NVIDIA information..."
nvidia-smi -q > "$RUN_DIR/nvidia-smi-q.txt" 2>&1 || true

echo "Collecting vLLM environment information..."
vllm collect-env > "$RUN_DIR/vllm-collect-env.txt" 2>&1 || true

echo "Collecting installed Python packages..."
python -m pip freeze > "$RUN_DIR/pip-freeze.txt"

echo
echo "Done."
echo "Environment saved under:"
echo "$RUN_DIR"