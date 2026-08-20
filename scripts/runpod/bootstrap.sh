#!/usr/bin/env bash
set -euo pipefail

# Bootstrap this repository on a RunPod PyTorch pod.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

echo "[bootstrap] repo: ${ROOT_DIR}"
echo "[bootstrap] commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

echo "[bootstrap] python: ${PYTHON_BIN} ($("${PYTHON_BIN}" --version))"

"${PYTHON_BIN}" -m pip install -U pip
"${PYTHON_BIN}" -m pip install -e .

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
echo "[bootstrap] next: ./scripts/runpod/check_gpu_env.sh"
