#!/usr/bin/env bash
set -euo pipefail

echo "Python:"
python --version

echo
echo "NVIDIA GPU:"
nvidia-smi || true

echo
echo "vLLM:"
python - <<'PY'
try:
    import vllm
    print(vllm.__version__)
except Exception as e:
    print("vLLM import failed:", e)
PY