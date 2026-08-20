#!/usr/bin/env bash
set -euo pipefail

# Multi-node DDP launcher for RunPod Instant Clusters.
# Run this on EVERY node after the cluster environment variables are present.
#
# Required/expected env (usually provided by RunPod Instant Clusters):
#   MASTER_ADDR / MASTER_PORT
#   NUM_NODES / NUM_TRAINERS / NODE_RANK
#   or WORLD_SIZE equivalents
#
# Usage:
#   ./scripts/runpod/run_ddp_multinode.sh --steps 20

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

MASTER_ADDR="${MASTER_ADDR:-${PRIMARY_ADDR:-}}"
MASTER_PORT="${MASTER_PORT:-${PRIMARY_PORT:-29500}}"
NUM_NODES="${NUM_NODES:-1}"
NUM_TRAINERS="${NUM_TRAINERS:-${NPROC_PER_NODE:-1}}"
NODE_RANK="${NODE_RANK:-0}"

if [[ -z "${MASTER_ADDR}" ]]; then
  echo "[multinode] ERROR: MASTER_ADDR/PRIMARY_ADDR is required"
  exit 1
fi

# RunPod Instant Clusters: bind NCCL to high-speed fabric, not eth0.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-ens1}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

OUT_DIR="${OUT_DIR:-results/phase1_runpod_multinode}"
mkdir -p "${OUT_DIR}"
METRICS_PATH="${OUT_DIR}/node${NODE_RANK}_trainers${NUM_TRAINERS}.json"

echo "[multinode] MASTER_ADDR=${MASTER_ADDR}"
echo "[multinode] MASTER_PORT=${MASTER_PORT}"
echo "[multinode] NUM_NODES=${NUM_NODES}"
echo "[multinode] NUM_TRAINERS=${NUM_TRAINERS}"
echo "[multinode] NODE_RANK=${NODE_RANK}"
echo "[multinode] NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nnodes "${NUM_NODES}" \
  --nproc_per_node "${NUM_TRAINERS}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  -m mini_training.train \
  --backend nccl \
  --metrics-path "${METRICS_PATH}" \
  "$@"

echo "[multinode] wrote ${METRICS_PATH}"
