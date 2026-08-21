#!/usr/bin/env bash
set -euo pipefail

# Compare TP training with and without --overlap on the same machine.
# Designed for RunPod multi-GPU (NCCL). Never invents numbers — only records
# what train.py prints / writes.
#
# Usage:
#   ./scripts/benchmark_tp_overlap.sh
#   OUT_DIR=results/phase8_overlap TP_SIZE=2 ./scripts/benchmark_tp_overlap.sh
#
# Optional env overrides:
#   TP_SIZE HIDDEN LAYERS HEADS BATCH SEQ STEPS WARMUP PROFILE_STEPS

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi

TP_SIZE="${TP_SIZE:-2}"
HIDDEN="${HIDDEN:-1024}"
LAYERS="${LAYERS:-8}"
HEADS="${HEADS:-16}"
BATCH="${BATCH:-4}"
SEQ="${SEQ:-512}"
STEPS="${STEPS:-30}"
WARMUP="${WARMUP:-3}"
PROFILE_STEPS="${PROFILE_STEPS:-6}"
OUT_DIR="${OUT_DIR:-results/phase8_overlap}"
mkdir -p "${OUT_DIR}"

COMMON_ARGS=(
  --steps "${STEPS}"
  --batch-size "${BATCH}"
  --seq-len "${SEQ}"
  --hidden-size "${HIDDEN}"
  --num-layers "${LAYERS}"
  --num-heads "${HEADS}"
  --mlp-ratio 4
  --dropout 0.0
  --log-interval 1
)

echo "[overlap-bench] commit=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[overlap-bench] out=${OUT_DIR}"
echo "[overlap-bench] tp=${TP_SIZE} hidden=${HIDDEN} layers=${LAYERS} heads=${HEADS} batch=${BATCH} seq=${SEQ} steps=${STEPS}"

run_one() {
  local tag="$1"
  shift
  local log="${OUT_DIR}/${tag}.log"
  local summary="${OUT_DIR}/${tag}_summary.json"
  echo "[overlap-bench] === ${tag} ==="
  set +e
  ./scripts/run_tp.sh "${TP_SIZE}" "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "${log}"
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ ${rc} -ne 0 ]]; then
    echo "[overlap-bench] FAILED ${tag} rc=${rc}" >&2
    return "${rc}"
  fi
  # Extract the last JSON summary line from train.py (rank0 prints one object).
  "${PYTHON_BIN}" - "${log}" "${summary}" <<'PY'
import json, sys
log_path, out_path = sys.argv[1], sys.argv[2]
last = None
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if line.startswith("{") and '"summary"' in line:
            last = line
if last is None:
    raise SystemExit(f"no summary JSON found in {log_path}")
obj = json.loads(last)
with open(out_path, "w") as f:
    json.dump(obj, f, indent=2)
    f.write("\n")
print(f"[overlap-bench] wrote {out_path}")
PY
}

# Throughput runs — no profiler (published tokens/s).
run_one "tp${TP_SIZE}_sync" 
run_one "tp${TP_SIZE}_overlap" --overlap

# Short profiled runs — qualitative only; do NOT use for tokens/s claims.
PROF_SYNC="${OUT_DIR}/profile_sync"
PROF_OV="${OUT_DIR}/profile_overlap"
mkdir -p "${PROF_SYNC}" "${PROF_OV}"
run_one "tp${TP_SIZE}_sync_profiled" \
  --steps "${PROFILE_STEPS}" \
  --profile-dir "${PROF_SYNC}" \
  --profile-wait 1 --profile-warmup 1 --profile-active 3
run_one "tp${TP_SIZE}_overlap_profiled" \
  --overlap \
  --steps "${PROFILE_STEPS}" \
  --profile-dir "${PROF_OV}" \
  --profile-wait 1 --profile-warmup 1 --profile-active 3

"${PYTHON_BIN}" - "${OUT_DIR}" "${TP_SIZE}" "${HIDDEN}" "${LAYERS}" "${HEADS}" "${BATCH}" "${SEQ}" "${STEPS}" "${WARMUP}" <<'PY'
import json, sys
from pathlib import Path

out = Path(sys.argv[1])
meta = {
    "tp_size": int(sys.argv[2]),
    "hidden": int(sys.argv[3]),
    "layers": int(sys.argv[4]),
    "heads": int(sys.argv[5]),
    "batch": int(sys.argv[6]),
    "seq": int(sys.argv[7]),
    "steps": int(sys.argv[8]),
    "warmup_note": "train.py discards first warmup steps internally; see each summary",
}

def load(tag):
    p = out / f"{tag}_summary.json"
    with open(p) as f:
        return json.load(f)

sync = load(f"tp{meta['tp_size']}_sync")
ov = load(f"tp{meta['tp_size']}_overlap")
s_sum, o_sum = sync["summary"], ov["summary"]

def speedup(a, b):
    if a is None or b is None or a == 0:
        return None
    return b / a

cmp = {
    "workload": meta,
    "sync": s_sum,
    "overlap": o_sum,
    "delta": {
        "step_time_ms_sync": s_sum.get("avg_step_time_ms"),
        "step_time_ms_overlap": o_sum.get("avg_step_time_ms"),
        "global_tok_s_sync": s_sum.get("avg_global_tokens_per_second"),
        "global_tok_s_overlap": o_sum.get("avg_global_tokens_per_second"),
        "tok_s_speedup_overlap_over_sync": speedup(
            s_sum.get("avg_global_tokens_per_second"),
            o_sum.get("avg_global_tokens_per_second"),
        ),
        "backward_ms_sync": s_sum.get("avg_backward_ms"),
        "backward_ms_overlap": o_sum.get("avg_backward_ms"),
    },
    "environment_sync": sync.get("environment"),
    "note": (
        "Profiled runs under profile_*/ are qualitative only; "
        "do not cite their tokens/s. Compare Chrome traces for tp_all_reduce vs aten::mm."
    ),
}
path = out / "overlap_comparison.json"
with open(path, "w") as f:
    json.dump(cmp, f, indent=2)
    f.write("\n")
print(json.dumps(cmp["delta"], indent=2))
print(f"[overlap-bench] wrote {path}")
PY

echo "[overlap-bench] done"
