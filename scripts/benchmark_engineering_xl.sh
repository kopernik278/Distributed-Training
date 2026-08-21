#!/usr/bin/env bash
set -euo pipefail

# XL engineering pass: larger model + WikiText-103 + full technique matrix.
# Thin wrapper around benchmark_engineering_suite.sh + analysis.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export SCALE="${SCALE:-xl}"
export DATASET="${DATASET:-wikitext103}"
export OUT_DIR="${OUT_DIR:-results/engineering_suite_xl}"
export STEPS="${STEPS:-30}"
export PROFILE="${PROFILE:-0}"

./scripts/benchmark_engineering_suite.sh

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

"${PYTHON_BIN}" "${ROOT_DIR}/scripts/analyze_engineering_suite.py" "${OUT_DIR}"
echo "[xl-suite] done → ${OUT_DIR}"
