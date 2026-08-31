#!/usr/bin/env bash
set -euo pipefail

echo "=== runtime ==="
python -c 'import vllm,torch; print("vLLM",vllm.__version__); print("torch",torch.__version__); print("torch_cuda",torch.version.cuda)'
nsys --version
sqlite3 --version
git --version

echo "=== storage ==="
df -h / /workspace

echo "=== gpu ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo "PREFLIGHT_OK"
