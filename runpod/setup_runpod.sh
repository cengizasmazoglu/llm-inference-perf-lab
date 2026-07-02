#!/usr/bin/env bash
set -euo pipefail

echo "Checking system..."
python --version || true
nvidia-smi || true

echo
echo "Upgrading pip..."
python -m pip install --upgrade pip

echo
echo "Installing Python packages..."
python -m pip install -r requirements.txt

echo
echo "Checking vLLM..."
vllm --help | head -40