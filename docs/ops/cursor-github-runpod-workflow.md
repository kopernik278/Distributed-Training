# Development Mode: Cursor + GitHub + RunPod

## Goal

Operate this repository like an AI Training Infra team:

```text
Cursor          → design, implement, review, document
GitHub          → source of truth, PR history, reproducibility
RunPod (GPU)    → real multi-GPU training, NCCL, profiling, benchmarks
```

Cursor Cloud / local CPU environments remain useful for:

- code scaffolding
- unit tests
- docs / RFC
- CPU smoke validation

They are **not** the place for portfolio-grade GPU performance claims.

## Responsibility split

| Layer | Tool | Owns |
|------|------|------|
| Design & coding | Cursor | architecture, RFC, implementation, tests, docs |
| Version control | GitHub | commits, PR, review trail, reproducible commit hash |
| GPU execution | RunPod | CUDA/NCCL runs, multi-GPU DDP/TP/PP, profiling artifacts |
| Analysis | Cursor + ChatGPT | bottleneck interpretation, next optimization |

## Canonical loop (enterprise style)

```text
1. Cursor: implement feature on branch
2. GitHub: commit + push + PR update
3. RunPod: git pull the same branch
4. RunPod: bootstrap env + GPU smoke test
5. RunPod: run distributed benchmark / training
6. RunPod: save metrics/profiles to results/
7. GitHub: commit experiment notes (not huge binary traces by default)
8. Cursor: analyze results and plan next change
```

## Branch policy

- Feature branch: `cursor/<topic>-ed3e`
- Base branch: `main`
- Every meaningful milestone must be pushed before asking RunPod to pull
- Every GPU experiment must record:
  - commit hash
  - GPU model / count
  - CUDA / PyTorch / NCCL versions
  - command line
  - metrics path

## What “enterprise-grade” means in this repo

We prioritize mainstream training infra practices:

1. `torchrun` launch
2. NCCL backend on GPU
3. Structured configs
4. Reproducible environment bootstrap
5. Correctness tests before optimization
6. Benchmark scripts with metadata
7. RFC before major subsystems (TP/PP/Checkpoint)
8. Clear separation of code machine vs GPU machine

Later stages may add:

- Megatron-LM / DeepSpeed for production-scale experiments
- Nsight Systems / Nsight Compute on RunPod
- Multi-node Instant Clusters
- WandB or local experiment index (optional)

## Current project stage under this mode

Project 1 Phase 1 (DDP baseline) is code-complete for CPU/Gloo validation.

Next execution priority on RunPod:

1. Single-node multi-GPU DDP smoke (`NCCL`)
2. 1/2/4 GPU scaling table with real tokens/s
3. Then Stage 3 Tensor Parallel on GPU
