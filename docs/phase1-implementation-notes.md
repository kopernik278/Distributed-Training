# Phase 1 Implementation Notes

This document is written as both engineering documentation and interview prep.
The goal is not only to run DDP, but to understand why the implementation is
structured this way and what real problems already appeared in the first
milestone.

## 1. Why start with synthetic data

The first bottleneck study should focus on the training system, not on data
loading or preprocessing noise. A random token workload keeps the signal clean:

- no filesystem variability,
- no tokenizer overhead,
- no dataloader worker tuning,
- easy control over sequence length and batch size.

This makes step time and throughput easier to reason about when comparing 1
process vs multiple processes.

## 2. Why use a minimal Transformer LM

Using a tiny Transformer already exercises the same categories of work that a
real LLM does:

- embedding lookup,
- self-attention,
- MLP,
- residual and normalization layers,
- next-token cross entropy.

That means the training loop is representative enough for later profiling, while
still staying small enough to debug.

## 3. Why DDP first

DDP is the cleanest baseline for distributed training:

1. each rank holds a full model replica,
2. each rank computes forward and backward on different input data,
3. gradients are synchronized through all-reduce during backward,
4. each optimizer step updates identical model weights on every rank.

This is the simplest place to build confidence in:

- process launch,
- rank/device mapping,
- gradient synchronization correctness,
- throughput accounting.

If this baseline is unstable, adding TP or PP only hides the root problem.

## 4. Current metric definitions

- `rank_tokens_per_second`: tokens processed by one rank divided by its local
  measured step time.
- `global_tokens_per_second`: sum of tokens across all ranks divided by the
  average step time across ranks.
- `step_time_ms`: average step time across ranks after synchronization.

These metrics are enough for the first scaling table:

| world size | batch size / rank | global batch | step time | global tok/s |
|------------|-------------------|--------------|-----------|--------------|

Later phases will add:

- communication time,
- compute/idle overlap,
- memory usage,
- MFU.

## 5. Real issues encountered in this phase

### Issue A: `python` command missing

The environment only exposed `python3`, while the launcher originally assumed
`python`.

**Fix**
- switch scripts and docs to `python3`.

**Why this matters**
- distributed launch scripts often fail for operational reasons before any model
  code runs; robust tooling is part of infra work.

### Issue B: `venv` creation failed because `ensurepip` was unavailable

The standard setup flow `python3 -m venv .venv` failed on this machine.

**Fix**
- fall back to `python3 -m pip install --break-system-packages ...`.

**Why this matters**
- training infra work is not only algorithmic; environment portability and
  bootstrap reliability are part of the job.

### Issue C: backend mismatch between configuration and runtime

The user-facing config default is `nccl`, but CPU-only validation must run on
`gloo`.

**Fix**
- runtime now records the actual backend used, instead of only the requested
  backend.

**Why this matters**
- if metrics are collected without recording the real backend, benchmark data can
  become misleading.

## 6. What this phase still does not solve

- no data parallel sharding dataset sampler yet,
- no activation checkpointing,
- no tensor/pipeline parallelism,
- no checkpoint save/resume,
- no profiling trace export.

Those omissions are deliberate. The current target is a trustworthy baseline,
not full feature coverage.
