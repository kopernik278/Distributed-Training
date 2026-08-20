#!/usr/bin/env bash
set -euo pipefail

# Single-node multi-GPU DDP launcher for RunPod.
# Usage:
#   ./scripts/runpod/run_ddp_gpu.sh            # auto use all visible GPUs
#   ./scripts/runpod/run_ddp_gpu.sh 2          # force 2 processes/GPUs
#   ./scripts/runpod/run_ddp_gpu.sh 2 --steps 20 --batch-size 8

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

if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
  NPROC_PER_NODE="$1"
  shift
else
  NPROC_PER_NODE="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 1)
PY
)"
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[run_ddp_gpu] ERROR: nproc_per_node must be >= 1"
  exit 1
fi

STEPS="${STEPS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEQ_LEN="${SEQ_LEN:-256}"
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-8}"
BACKEND="${BACKEND:-nccl}"
OUT_DIR="${OUT_DIR:-results/phase1_runpod_gpu}"
mkdir -p "${OUT_DIR}"

METRICS_PATH="${OUT_DIR}/ws${NPROC_PER_NODE}.json"

echo "[run_ddp_gpu] nproc_per_node=${NPROC_PER_NODE}"
echo "[run_ddp_gpu] backend=${BACKEND}"
echo "[run_ddp_gpu] metrics=${METRICS_PATH}"

# Helpful NCCL defaults for single-node bring-up.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

if [[ "${NPROC_PER_NODE}" -eq 1 ]]; then
  "${PYTHON_BIN}" -m mini_training.train \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --num-layers "${NUM_LAYERS}" \
    --num-heads "${NUM_HEADS}" \
    --backend "${BACKEND}" \
    --metrics-path "${METRICS_PATH}" \
    "$@"
else
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "${NPROC_PER_NODE}" \
    -m mini_training.train \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --seq-len "${SEQ_LEN}" \
    --hidden-size "${HIDDEN_SIZE}" \
    --num-layers "${NUM_LAYERS}" \
    --num-heads "${NUM_HEADS}" \
    --backend "${BACKEND}" \
    --metrics-path "${METRICS_PATH}" \
    "$@"
fi

echo "[run_ddp_gpu] wrote ${METRICS_PATH}"
