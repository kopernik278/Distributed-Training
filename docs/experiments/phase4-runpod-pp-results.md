# Phase 4 Measured Results (RunPod / Pipeline Parallel)

> Measured on real multi-GPU RunPod hardware. Numbers are not fabricated.

## Hardware / software

- GPU: 2 x NVIDIA GeForce RTX 3090
- Backend: `nccl`
- PyTorch: `2.8.0+cu128`
- CUDA: `12.8`
- NCCL: `2.27.3`
- Commit: `2ecb7ab`
- Pod: community cloud, ~$0.44/hr (stopped after measurement)

## Correctness (before GPU timing)

On the same pod host, multiprocess Gloo numeric test passed:

```text
test_pp2_matches_dense_loss_and_grads ... ok
```

PP=2 1F1B loss/grads match an **untied** dense `MiniTransformerLM` on identical microbatches
(`loss_err < 1e-5`, `max_grad_err < 1e-4`).

## Workload

| knob | pp=1 baseline | pp=2 1F1B |
|---|---:|---:|
| GPUs / world_size | 1 | 2 |
| microbatch size | 8 | 8 |
| num_microbatches | 1 | 4 |
| tokens / optimizer step | 2048 | 8192 |
| seq_len | 256 | 256 |
| hidden / layers / heads | 256 / 4 / 8 | 256 / 4 / 8 |
| steps (discard) | 20 (2) | 20 (2) |
| dropout | 0 | 0 |

## Summary

| metric | pp=1 (1×3090) | pp=2 (2×3090, M=4) |
|---|---:|---:|
| avg loss | 8.3610 | 8.3513 |
| avg step time (ms) | 9.735 | 24.946 |
| avg global tok/s | 210428 | 329066 |

Raw JSON: `phase4_runpod_pp1_summary.json`, `phase4_runpod_pp2_summary.json`.

## How to read these numbers

1. **PP step processes more tokens** when `num_microbatches=4` (8192 vs 2048).  
   Compare throughput, not raw step time alone.
2. Approximate same-work view: 4× pp1 steps ≈ `4 × 9.735 ≈ 38.9 ms` for 8192 tokens  
   (~211k tok/s). PP2 finishes that work in ~24.9 ms (~329k tok/s) → ~**1.56×** wall speedup on 2 GPUs.
3. On this **tiny** model, PP will not look like production Megatron scaling:
   P2P + pipeline bubble dominate; value here is **correct NCCL path + measured baseline**.
4. `backward_ms` is reported as `0` for PP because the engine times the whole 1F1B schedule in the forward bucket (F/B interleaved).

## Notes

- End-to-end CUDA/NCCL PP + 1F1B path validated.
- Numeric correctness validated separately (dense vs PP multiprocess).
- Do not claim PP > DDP/TP from this tiny config without a larger-layer study.
