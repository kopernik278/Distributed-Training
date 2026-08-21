# Distributed-Training

Mini LLM training infrastructure project for AI Infra / Distributed Training
Engineer interviews.

If you are **not** already an infra engineer, start here:
[`docs/learn/BEGINNER_GUIDE.md`](./docs/learn/BEGINNER_GUIDE.md)

Long-term learning context: [`AI_INFRA_CONTEXT.md`](./AI_INFRA_CONTEXT.md)

## Development mode (canonical)

```text
Cursor  +  GitHub  +  RunPod
```

## Current milestone

**Project 1 / Phase 6: Profiling (Chrome traces + comm labels)**

Completed Phases 1–5 (DDP, TP, DP×TP, Pipeline 1F1B, Checkpoint) plus profiler hooks.

## Tests

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

## Profile a short run (CPU/Gloo is fine for learning)

```bash
BACKEND=gloo ./scripts/run_tp.sh 1 --steps 4 --batch-size 2 --seq-len 32 \
  --hidden-size 32 --num-heads 4 --dropout 0.0 \
  --profile-dir /tmp/prof_demo --profile-wait 1 --profile-warmup 1 --profile-active 2
```

Open `/tmp/prof_demo/rank0.json` in [Perfetto](https://ui.perfetto.dev).
Do not treat profiled tokens/s as a benchmark.

## Next stages

1. Communication/computation overlap
2. Optional Adam-state TP reshard / PP consolidate
3. Optional larger-model multi-GPU studies
