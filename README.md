# Distributed-Training

Mini LLM training infrastructure project for AI Infra / Distributed Training
Engineer interviews.

Long-term learning context: [`AI_INFRA_CONTEXT.md`](./AI_INFRA_CONTEXT.md)

## Development mode (canonical)

```text
Cursor  +  GitHub  +  RunPod
```

- **Cursor**: design, implement, test scaffolding, docs/RFC
- **GitHub**: source of truth and reproducible commit history
- **RunPod**: real GPU / multi-GPU / NCCL execution and profiling

## Current milestone

**Project 1 / Phase 4: Pipeline Parallel + 1F1B**

Completed:

- Phase 1 DDP baseline + real 2x RTX 4090 NCCL measurements
- Phase 2 Megatron-style Tensor Parallel
- Phase 3 DP×TP 2D parallel
- Phase 4 Pipeline stages + educational 1F1B engine (async P2P)

## Repository layout

```text
src/mini_training/
  parallel_state.py   # dp × pp × tp groups
  mappings.py / layers.py
  model.py            # MiniTransformerLM + PipelineStage
  pipeline.py         # 1F1B engine
  train.py
scripts/
  run_tp.sh / run_dp_tp.sh / run_pp.sh
docs/
  design/RFC-001 … RFC-004
  phase2-tensor-parallel.md
  phase3-dp-tp.md
  phase4-pipeline-parallel.md
```

## Install

```bash
python3 -m pip install -e .
```

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Run

### Phase 1 style DDP

```bash
NPROC_PER_NODE=2 ./scripts/run_single_node_ddp.sh --steps 10 --batch-size 4 --seq-len 128
```

### Phase 2 Tensor Parallel

```bash
./scripts/run_tp.sh 2 --steps 5 --batch-size 2 --seq-len 64 --hidden-size 64 --num-heads 8
```

### Phase 3 DP × TP

```bash
./scripts/run_dp_tp.sh 2 2 --steps 5 --batch-size 2 --seq-len 32 \
  --hidden-size 64 --num-layers 2 --num-heads 8 --dropout 0.0
```

### Phase 4 Pipeline Parallel (1F1B)

```bash
./scripts/run_pp.sh 2 --steps 3 --batch-size 2 --seq-len 16 \
  --hidden-size 32 --num-layers 4 --num-heads 4 --num-microbatches 4 --dropout 0.0
```

## Important rule

Never report fabricated performance numbers. CPU/Gloo is for correctness; RunPod CUDA/NCCL is for portfolio measurements.

## Next stages

1. Distributed Checkpoint save/resume/reshard
2. Profiling hooks and communication/computation overlap
3. Optional: RunPod multi-GPU PP / 3D-parallel NCCL metrics
