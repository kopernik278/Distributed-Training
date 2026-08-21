# Phase 6: Profiling (see compute vs communication)

## What landed

- RFC: `docs/design/RFC-006-profiling.md`
- `src/mini_training/profiler.py`: optional `torch.profiler` + Chrome export
- Named ranges: `train_step`, `tp_all_reduce`, `tp_all_gather`, `pp_*`, `optimizer_step`
- CLI: `--profile-dir`, `--profile-wait`, `--profile-warmup`, `--profile-active`
- Tests: `tests/test_profiler.py`
- Beginner walkthrough: `docs/learn/BEGINNER_GUIDE.md`

## How to run

```bash
BACKEND=gloo ./scripts/run_tp.sh 1 --steps 4 --batch-size 2 --seq-len 32 \
  --hidden-size 32 --num-heads 4 --dropout 0.0 \
  --profile-dir /tmp/prof_demo --profile-wait 1 --profile-warmup 1 --profile-active 2
```

Then:

1. Read `/tmp/prof_demo/rank0_summary.txt` (operator table)
2. Open `/tmp/prof_demo/rank0.json` in https://ui.perfetto.dev

On GPU (do **not** mix with published tokens/s):

```bash
BACKEND=nccl ./scripts/run_tp.sh 2 --steps 8 ... --profile-dir profiles/tp2
```

Look for `tp_all_reduce` next to `aten::mm`. Sequential bars ⇒ no overlap yet.

Phase 7 implements opt-in overlap: `docs/phase7-comm-overlap.md`.

## Next

- Delayed wait for RowParallel forward AllReduce
- Optional Nsight Systems on RunPod
