# Phase 1: DDP Baseline

## Why this phase exists

The first milestone is a trustworthy distributed training baseline. Before adding
tensor parallelism, pipeline parallelism, or checkpoint resharding, we need a
training loop that can:

1. launch cleanly on 1 or N processes,
2. synchronize gradients correctly with DDP,
3. report stable performance metrics,
4. stay simple enough to profile and debug.

This is the foundation for every later optimization story in interview settings.

## Scope

- Synthetic next-token prediction workload
- Minimal Transformer language model
- PyTorch DDP baseline
- Single-node launch script
- Metrics logging:
  - loss
  - step time
  - forward / backward / optimizer time
  - rank tokens/s
  - global tokens/s
- Correctness tests
- Scaling benchmark script

## Interview relevance

This phase prepares you to explain:

- what DDP synchronizes and when,
- why global batch size scales with world size,
- how step time changes from 1 GPU to N GPUs,
- how to separate compute bottlenecks from communication bottlenecks,
- why warm-up discard and environment metadata matter for experiment quality.

## Next phases

Phase 2 will add tensor parallel linear layers and collective patterns
(all-gather / reduce-scatter).
