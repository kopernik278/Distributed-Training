# Distributed-Training

Mini LLM training infrastructure project for AI Infra / Distributed Training
Engineer interviews.

If you are **not** already an infra engineer, start here:
[`docs/learn/BEGINNER_GUIDE.md`](./docs/learn/BEGINNER_GUIDE.md)

## Current milestone

**Project 1 / Phase 7: Communication / computation overlap**

Phases 1–6 remain: DDP, TP, DP×TP, Pipeline 1F1B, Checkpoint, Profiler.

## Tests

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

## Overlap (opt-in)

```bash
BACKEND=gloo ./scripts/run_tp.sh 2 --steps 3 --hidden-size 64 --num-heads 8 --overlap
```

ColumnParallel backward launches `AllReduce(dX)` and computes `dW` before waiting.
Tiny CPU models will not speed up; use GPU traces to see NCCL under GEMM.

## Next stages

1. Hide RowParallel forward AllReduce behind the next layer
2. Optional Adam-state TP reshard / PP consolidate
3. Optional larger-model multi-GPU overlap measurements
