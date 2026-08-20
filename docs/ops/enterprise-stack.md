# Enterprise Training Stack (Current Target)

This repository adopts a mainstream AI Training Infra stack, introduced gradually.

## Platform triad

```text
Cursor  +  GitHub  +  RunPod
 code      source     GPU runtime
```

## Framework / tool progression

### Now (Phase 1 bring-up)

| Concern | Choice | Why |
|--------|--------|-----|
| Training framework | PyTorch | industry default for research+infra work |
| Launch | `torchrun` | standard process orchestration |
| Collective backend | NCCL (GPU) / Gloo (CPU fallback) | production GPU communication |
| Parallelism (current) | DDP | first correct distributed baseline |
| Config | CLI + YAML defaults | reproducible experiments |
| Metrics | JSON structured logs | commit-addressable artifacts |
| Remote GPU | RunPod Pods | fast rentable CUDA boxes |

### Next (Project 1 Stage 3+)

| Concern | Choice | Why |
|--------|--------|-----|
| Tensor / Pipeline parallel | custom mini-engine first, then Megatron-LM lab | learn internals before framework magic |
| Profiling | PyTorch Profiler → Nsight Systems → Nsight Compute | standard perf workflow |
| Checkpoint | PyTorch Distributed Checkpoint APIs | production resume/reshard skill |
| Multi-node | RunPod Instant Clusters | real inter-node NCCL path |

### Later portfolio extensions

- DeepSpeed ZeRO experiments (as comparison lab, not first implementation)
- Megatron-LM performance optimization project
- NCCL/RoCE benchmarks on multi-node fabric
- Triton/CUDA kernels
- MoE expert parallel

## Non-negotiables

1. Correctness before speed.
2. No fabricated GPU numbers.
3. Every GPU result includes hardware + commit metadata.
4. Code changes land in GitHub before RunPod pulls them.
5. Stop GPU pods when idle.
