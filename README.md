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

Workflow docs:

- [`docs/ops/cursor-github-runpod-workflow.md`](./docs/ops/cursor-github-runpod-workflow.md)
- [`docs/ops/runpod-setup-guide.md`](./docs/ops/runpod-setup-guide.md)
- [`docs/ops/enterprise-stack.md`](./docs/ops/enterprise-stack.md)

## Current milestone

**Project 1 / Phase 1 (Stage 1-2): DDP baseline**

- minimal Transformer language model
- synthetic next-token workload
- PyTorch DDP training loop
- forward / backward / optimizer timing breakdown
- correctness tests
- DDP scaling benchmark script
- RunPod GPU launch scripts
- RFC + experiment documentation

## Repository layout

```text
AI_INFRA_CONTEXT.md
configs/
  phase1_gpu_smoke.yaml
src/mini_training/
scripts/
  run_single_node_ddp.sh
  benchmark_ddp_scaling.sh
  runpod/
    bootstrap.sh
    check_gpu_env.sh
    run_ddp_gpu.sh
    run_ddp_multinode.sh
tests/
docs/
  design/
  experiments/
  ops/
requirements/
  gpu.txt
```

## Install (CPU / Cursor machine)

```bash
python3 -m pip install -e .
```

CPU smoke:

```bash
python3 -m mini_training.train --steps 5 --batch-size 2 --seq-len 64
python3 -m unittest discover -s tests -v
```

## Run on RunPod (GPU)

Follow [`docs/ops/runpod-setup-guide.md`](./docs/ops/runpod-setup-guide.md), then:

```bash
./scripts/runpod/bootstrap.sh
./scripts/runpod/check_gpu_env.sh
./scripts/runpod/run_ddp_gpu.sh 1
./scripts/runpod/run_ddp_gpu.sh 2
```

Scaling benchmark on GPU:

```bash
WORLD_SIZES=1,2 BACKEND=nccl STEPS=20 \
OUT_DIR=results/phase1_runpod_gpu \
./scripts/benchmark_ddp_scaling.sh
```

## Metrics

Each step logs:

- `loss`
- `step_time_ms`
- `forward_ms`
- `backward_ms`
- `optimizer_ms`
- `rank_tokens_per_second`
- `global_tokens_per_second`

Metrics files also include environment metadata (device, backend, commit, GPU name when available).

## Important rule

Never report fabricated performance numbers.

- CPU/Gloo results: instrumentation only
- RunPod CUDA/NCCL results: valid portfolio measurements (with hardware metadata)

## Next stages (Project 1)

1. RunPod single-node NCCL DDP baseline table
2. Tensor Parallel Linear (`ColumnParallel` / `RowParallel`)
3. Pipeline Parallel + 1F1B
4. Distributed Checkpoint save/resume/reshard
5. Profiling hooks and communication/computation overlap
