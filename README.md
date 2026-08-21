# Distributed-Training

Mini LLM training infrastructure project for AI Infra / Distributed Training
Engineer interviews.

If you are **not** already an infra engineer, start here:
[`docs/learn/BEGINNER_GUIDE.md`](./docs/learn/BEGINNER_GUIDE.md)

## Current milestone

**Project 1 / Phase 9: Sequence Parallel + Vocab Parallel**

Phases 1–8: DDP, TP, DP×TP, Pipeline 1F1B, Checkpoint, Profiler, Comm overlap,
delayed RowParallel wait (+ RunPod overlap measurement).

## Tests

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

## Sequence / Vocab parallel (opt-in)

```bash
BACKEND=gloo ./scripts/run_tp.sh 2 --steps 3 --seq-len 64 --hidden-size 64 \
  --num-heads 8 --sequence-parallel --vocab-parallel
```

Requires `tp>1`, `seq_len % tp == 0`, and for VP `vocab_size % tp == 0`.

## Next stages

1. Optional Adam-state TP reshard / PP consolidate
2. Optional RunPod memory footprint comparison with `--sequence-parallel`

Overlap GPU measurement: [`docs/experiments/phase8-runpod-overlap-results.md`](./docs/experiments/phase8-runpod-overlap-results.md).
