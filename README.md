# Distributed-Training

Mini LLM training infrastructure project for AI Infra / Distributed Training
Engineer interviews.

If you are **not** already an infra engineer, start here:
[`docs/learn/BEGINNER_GUIDE.md`](./docs/learn/BEGINNER_GUIDE.md)

## Current milestone

**Project 1 / Phase 8: Delayed RowParallel forward AllReduce wait**

Phases 1–7 remain: DDP, TP, DP×TP, Pipeline 1F1B, Checkpoint, Profiler,
ColumnParallel backward overlap.

## Tests

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

## Overlap (opt-in)

```bash
BACKEND=gloo ./scripts/run_tp.sh 2 --steps 3 --hidden-size 64 --num-heads 8 --overlap
```

Under `--overlap`:

1. ColumnParallel backward: `AllReduce(dX)` overlaps `dW` GEMM
2. RowParallel forward: AllReduce wait is delayed until `finalize` / residual
   (`skip_bias_add` + flush)

Tiny CPU models will not speed up; use GPU traces to see NCCL under GEMM.

## Next stages

1. Optional Adam-state TP reshard / PP consolidate
2. Optional larger-model multi-GPU overlap measurements
3. Sequence / vocab parallel
