# Distributed-Training

Mini LLM training infrastructure project for AI Infra / Distributed Training
Engineer interviews.

If you are **not** already an infra engineer, start here:
[`docs/learn/BEGINNER_GUIDE.md`](./docs/learn/BEGINNER_GUIDE.md)

## Current milestone

**Project 1 complete (Phases 1–9) + engineering enhancement (base + XL)**

DDP, TP, DP×TP, Pipeline 1F1B, Checkpoint, Profiler, Comm overlap, delayed
RowParallel wait, Sequence/Vocab Parallel — validated on RunPod with WikiText-2
(~160M) and WikiText-103 (~479M XL).

**Final report:** [`docs/reports/FINAL_ENGINEERING_REPORT.md`](./docs/reports/FINAL_ENGINEERING_REPORT.md)

## Tests

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

## Engineering GPU suites

```bash
# Base (~160M, WikiText-2)
GPU_COUNT=2 ./scripts/benchmark_engineering_suite.sh

# XL (~479M, WikiText-103)
./scripts/benchmark_engineering_xl.sh
```

Committed samples: `docs/experiments/engineering_suite/`, `docs/experiments/engineering_suite_xl/`.

## Sequence / Vocab parallel (opt-in)

```bash
BACKEND=gloo ./scripts/run_tp.sh 2 --steps 3 --seq-len 64 --hidden-size 64 \
  --num-heads 8 --sequence-parallel --vocab-parallel
```

Requires `tp>1`, `seq_len % tp == 0`, and for VP `vocab_size % tp == 0`.

## Next stages (optional)

1. Multi-node / NVLink hosts for positive DDP scaling and eng-seq PP
2. Adam-state TP reshard / PP consolidate
3. Activation-memory comparison with `--sequence-parallel`

Overlap GPU measurement: [`docs/experiments/phase8-runpod-overlap-results.md`](./docs/experiments/phase8-runpod-overlap-results.md).
