#!/usr/bin/env bash
set -euo pipefail

# Launch pure tensor-parallel training (world_size == TP size).
# Usage:
#   ./scripts/run_tp.sh 2 --steps 10 --batch-size 4

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <tp_size> [train args...]"
  exit 1
fi

TP_SIZE="$1"
shift

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]] && "${ROOT_DIR}/.venv/bin/python" -c "import torch, mini_training" >/dev/null 2>&1; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python
fi

BACKEND="${BACKEND:-nccl}"
if ! "${PYTHON_BIN}" - <<'PY'
import torch, sys
sys.exit(0 if torch.cuda.is_available() else 1)
PY
then
  BACKEND="gloo"
fi

echo "[run_tp] tp_size=${TP_SIZE} backend=${BACKEND} python=${PYTHON_BIN}"

if [[ "${TP_SIZE}" -eq 1 ]]; then
  "${PYTHON_BIN}" -m mini_training.train --tensor-parallel-size 1 --backend "${BACKEND}" "$@"
else
  "${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes 1 \
    --nproc_per_node "${TP_SIZE}" \
    -m mini_training.train \
    --tensor-parallel-size "${TP_SIZE}" \
    --backend "${BACKEND}" \
    "$@"
fi
