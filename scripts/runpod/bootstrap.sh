#!/usr/bin/env bash
set -euo pipefail

# Bootstrap this repository on a RunPod PyTorch pod.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[bootstrap] repo: ${ROOT_DIR}"
echo "[bootstrap] commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

BASE_PYTHON="${PYTHON_BIN:-python3}"
if ! command -v "${BASE_PYTHON}" >/dev/null 2>&1; then
  BASE_PYTHON=python
fi

# Prefer a local venv so we avoid PEP 668 system-python install blocks on
# modern RunPod images, while still inheriting the template's CUDA torch.
VENV_DIR="${ROOT_DIR}/.venv"
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "[bootstrap] creating venv with system site packages (for CUDA torch)"
  "${BASE_PYTHON}" -m venv --system-site-packages "${VENV_DIR}"
fi

PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

echo "[bootstrap] python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version))"

"${PIP_BIN}" install -U pip
"${PIP_BIN}" install -e .
"${PIP_BIN}" install -q pyarrow

mkdir -p results profiles checkpoints configs

"${PYTHON_BIN}" - <<'PY'
import platform
import torch

print("[bootstrap] environment")
print("  platform:", platform.platform())
print("  python:", platform.python_version())
print("  torch:", torch.__version__)
print("  cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("  cuda:", torch.version.cuda)
    print("  gpu_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  gpu[{i}]:", torch.cuda.get_device_name(i))
    if hasattr(torch.cuda, "nccl"):
        try:
            print("  nccl:", ".".join(str(x) for x in torch.cuda.nccl.version()))
        except Exception as exc:  # noqa: BLE001
            print("  nccl: unavailable", exc)
PY

echo "[bootstrap] done"
echo "[bootstrap] activate with: source .venv/bin/activate"
echo "[bootstrap] next: ./scripts/runpod/check_gpu_env.sh"
