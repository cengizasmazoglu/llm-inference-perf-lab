#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${OUT_DIR:-/tmp/nsys_cuda_smoke}"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

if [[ -n "${NSYS_BIN:-}" ]]; then
  NSYS="$NSYS_BIN"
else
  NSYS="$(command -v nsys || true)"
fi

if [[ -z "$NSYS" ]]; then
  echo "ERROR: nsys not found."
  exit 1
fi

echo "========================================"
echo "Nsight CUDA smoke test"
echo "========================================"

"$NSYS" --version

echo
echo "PyTorch CUDA environment:"

python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))
PY

cat > "$OUT_DIR/workload.py" <<'PY'
import torch

device = "cuda"

a = torch.randn(
    (4096, 4096),
    device=device,
    dtype=torch.float16,
)

b = torch.randn(
    (4096, 4096),
    device=device,
    dtype=torch.float16,
)

# Warm up CUDA context.
for _ in range(3):
    c = a @ b

torch.cuda.synchronize()

# Known-active CUDA workload.
for _ in range(20):
    c = a @ b

torch.cuda.synchronize()

print(float(c[0, 0]))
PY

echo
echo "Collecting smoke trace..."

"$NSYS" profile \
  --trace=cuda \
  --sample=none \
  --force-overwrite=true \
  --output="$OUT_DIR/torch_cuda_smoke" \
  python "$OUT_DIR/workload.py"

REPORT="$OUT_DIR/torch_cuda_smoke.nsys-rep"
STATS="$OUT_DIR/kernel-stats.txt"

if [[ ! -f "$REPORT" ]]; then
  echo "ERROR: Nsight report was not created."
  exit 1
fi

echo
echo "Checking captured CUDA kernels..."

"$NSYS" stats \
  --report cuda_gpu_kern_sum \
  "$REPORT" \
  > "$STATS"

cat "$STATS"

if grep -qi \
  'does not contain CUDA kernel data' \
  "$STATS"; then

  echo
  echo "ERROR: Nsight produced a report but captured no CUDA kernels."
  exit 1
fi

if ! grep -q \
  'CUDA GPU Kernel Summary' \
  "$STATS"; then

  echo
  echo "ERROR: CUDA kernel summary was not found."
  exit 1
fi

echo
echo "========================================"
echo "NSIGHT CUDA SMOKE TEST PASSED"
echo "========================================"