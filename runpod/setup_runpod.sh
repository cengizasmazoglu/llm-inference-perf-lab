#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv-runpod}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

VLLM_VERSION="0.27.1"
CUDA_WHEEL="cu129"

VLLM_WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_WHEEL}-cp38-abi3-manylinux_2_28_x86_64.whl"

PYTORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_WHEEL}"

cd "$REPO_ROOT"

echo "========================================"
echo "RunPod Lab Setup"
echo "========================================"
echo "Repository: $REPO_ROOT"
echo "Virtual environment: $VENV_DIR"
echo "Python target: $PYTHON_VERSION"
echo "vLLM: $VLLM_VERSION+$CUDA_WHEEL"
echo

echo "=== GPU ==="
nvidia-smi -L
echo

echo "=== NVIDIA DRIVER ==="
nvidia-smi
echo

echo "=== INSTALLING UV ==="
python -m pip install --upgrade uv
echo
uv --version
echo

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "=== CREATING CLEAN PYTHON ENVIRONMENT ==="

  uv venv "$VENV_DIR" \
    --python "$PYTHON_VERSION" \
    --managed-python \
    --seed
else
  echo "=== REUSING EXISTING ENVIRONMENT ==="
  echo "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo
echo "=== PYTHON ==="
python --version
which python
echo

echo "=== INSTALLING CUDA-MATCHED vLLM ==="

uv pip install \
  "$VLLM_WHEEL_URL" \
  --extra-index-url "$PYTORCH_INDEX_URL" \
  --index-strategy unsafe-best-match

echo
echo "=== INSTALLING LAB DEPENDENCIES ==="

uv pip install \
  -r requirements.txt \
  --index-strategy unsafe-best-match

echo
echo "=== INSTALLED VERSIONS ==="

python - <<'PY'
import torch
import vllm
import huggingface_hub

print("vLLM:", vllm.__version__)
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("Hugging Face Hub:", huggingface_hub.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)

        print(f"GPU {i}: {props.name}")
        print(
            f"GPU {i} VRAM: "
            f"{props.total_memory / (1024 ** 3):.2f} GiB"
        )
PY

echo
echo "=== vLLM CLI ==="
vllm --version

echo
echo "========================================"
echo "Setup complete."
echo
echo "Activate with:"
echo
echo "  source .venv-runpod/bin/activate"
echo
echo "Then run:"
echo
echo "  bash vllm/scripts/03_run_baseline.sh"
echo "========================================"