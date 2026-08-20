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

**Project 1 / Phase 3: DP × TP (2D parallel)**

Completed:

- Phase 1 DDP baseline + real 2x RTX 4090 NCCL measurements
- Phase 2 Megatron-style `ColumnParallelLinear` / `RowParallelLinear`
- Phase 3 DP×TP: seeded init, DP broadcast, DDP-on-DP-group, `dp×tp` launcher

## Repository layout

```text
src/mini_training/
  config.py
  data.py
  distributed.py
  parallel_state.py
  mappings.py
  layers.py
  model.py
  train.py
scripts/
  run_tp.sh
  run_dp_tp.sh
  benchmark_ddp_scaling.sh
  runpod/
docs/
  design/RFC-001-phase1-ddp-baseline.md
  design/RFC-002-tensor-parallel-linear.md
  design/RFC-003-dp-tp-2d-parallel.md
  phase2-tensor-parallel.md
  phase3-dp-tp.md
  experiments/
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
./scripts/run_tp.sh 1 --steps 5 --batch-size 2 --seq-len 64
./scripts/run_tp.sh 2 --steps 5 --batch-size 2 --seq-len 64 --hidden-size 64 --num-heads 8
```

### Phase 3 DP × TP

```bash
./scripts/run_dp_tp.sh 2 2 --steps 5 --batch-size 2 --seq-len 32 \
  --hidden-size 64 --num-layers 2 --num-heads 8 --dropout 0.0
```

### RunPod GPU

```bash
./scripts/runpod/bootstrap.sh
./scripts/runpod/check_gpu_env.sh
BACKEND=nccl ./scripts/run_tp.sh 2 --steps 20 --batch-size 8 --seq-len 256 \
  --hidden-size 256 --num-layers 4 --num-heads 8 --metrics-path results/tp2.json
BACKEND=nccl ./scripts/run_dp_tp.sh 2 2 --steps 20 --batch-size 8 --seq-len 256 \
  --hidden-size 256 --num-layers 4 --num-heads 8 --metrics-path results/dp2_tp2.json
```

## Important rule

Never report fabricated performance numbers. CPU/Gloo is for correctness; RunPod CUDA/NCCL is for portfolio measurements.

## Next stages

1. Pipeline Parallel + 1F1B
2. Distributed Checkpoint save/resume/reshard
3. Profiling hooks and communication/computation overlap
4. Optional: RunPod 4-GPU DP×TP NCCL metrics when hardware is available
