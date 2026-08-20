#!/usr/bin/env bash
set -euo pipefail

# Validate GPU / NCCL readiness on RunPod before training claims.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  CANDIDATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.venv/bin/python"
  if [[ -x "${CANDIDATE}" ]] && "${CANDIDATE}" -c "import torch, mini_training" >/dev/null 2>&1; then
    PYTHON_BIN="${CANDIDATE}"
  else
    PYTHON_BIN=python3
  fi
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

echo "[check] nvidia-smi"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "[check] ERROR: nvidia-smi not found"
  exit 1
fi

echo "[check] torch/cuda/nccl"
"${PYTHON_BIN}" - <<'PY'
import os
import sys
import torch

ok = True
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available inside PyTorch")
    ok = False
else:
    print("cuda:", torch.version.cuda)
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"gpu[{i}]:", torch.cuda.get_device_name(i))
        major, minor = torch.cuda.get_device_capability(i)
        print(f"capability[{i}]: {major}.{minor}")

    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = torch.matmul(x, x)
        torch.cuda.synchronize()
        print("matmul_smoke: ok", float(y[0, 0]))
    except Exception as exc:  # noqa: BLE001
        print("ERROR: CUDA matmul smoke failed:", exc)
        ok = False

    if hasattr(torch.cuda, "nccl"):
        try:
            print("nccl:", ".".join(str(v) for v in torch.cuda.nccl.version()))
        except Exception as exc:  # noqa: BLE001
            print("WARNING: cannot query NCCL version:", exc)
    else:
        print("WARNING: torch.cuda.nccl missing")

# Useful launch metadata
for key in [
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NUM_NODES",
    "NUM_TRAINERS",
    "NODE_RANK",
]:
    if key in os.environ:
        print(f"env[{key}]={os.environ[key]}")

sys.exit(0 if ok else 1)
PY

echo "[check] GPU environment looks ready"
