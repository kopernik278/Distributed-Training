# RFC-006: Profiling Hooks (Compute vs Communication)

## Motivation

Phases 1–5 made the engine *correct*. Interviewers next ask:

> How do you know the GPU is computing versus waiting on NCCL?

This milestone adds a **PyTorch Profiler** path that writes Chrome traces and a
tiny **comm vs compute** summary JSON. Overlap *tuning* (hiding AllReduce behind
GEMM) is a follow-up; first we must *see* the timeline.

## Goals

1. `--profile-dir` exports Chrome traces (`chrome://tracing` or Perfetto).
2. Record CPU + CUDA activities when CUDA is available.
3. Annotate training steps with `record_function("train_step")`.
4. Annotate TP collectives (`tp_all_reduce`, `tp_all_gather`) and PP P2P.
5. Write a compact `profile_summary.json` (self-time of named ranges when present).
6. Tests: profiler context is a no-op when disabled; enabled run writes a file.

## Non-goals

- Production Nsight Systems / NCCL flight recorder integration
- Automatic overlap of TP AllReduce with the next GEMM
- Changing numerical results when profiling is off

## How to read a trace

1. Open `rank0.json` in https://ui.perfetto.dev
2. Look for `aten::mm` / `aten::addmm` (compute) vs `nccl:*` / `tp_all_reduce` (comm)
3. If comm bars sit *after* compute with a gap, the GPU was idle (no overlap)

## Risks

- Profiler overhead distorts tokens/s — never mix profiled steps into published benchmarks
- First-step CUDA warmup pollutes traces — skip first N steps via `--profile-wait`

## Future Work

- Overlap DDP AllReduce with backward of later layers (bucket)
- Overlap TP reduce with next-layer compute
- NVTX export for Nsight
