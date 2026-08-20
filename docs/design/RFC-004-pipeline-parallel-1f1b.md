# RFC-004: Pipeline Parallel + 1F1B

## Motivation

DP replicates a model; TP shards matrices inside a layer.
**Pipeline Parallel (PP)** shards **layers across ranks** and streams
**microbatches** through stages with point-to-point activation traffic.

Without a good schedule, PP bubbles waste compute. This milestone implements
the classic **1F1B** (one-forward-one-backward) schedule used by Megatron-LM.

## Goals

1. Extend rank layout to `world_size = dp × pp × tp` (Megatron ordering).
2. Split Transformer blocks across pipeline stages (`num_layers % pp == 0`).
3. Stage-0 owns embeddings; last stage owns final norm + LM head + loss.
4. P2P send/recv of activations / gradients between adjacent PP ranks.
5. Implement warmup → steady 1F1B → cooldown.
6. Keep `pp=1` path identical to Phase 3 (DP×TP).
7. CPU/Gloo correctness tests for `pp=2`.

## Non-goals

- Interleaved / virtual pipeline stages
- Asynchronous P2P overlap tuning
- Activation checkpointing inside stages
- Full 3D DP×PP×TP GPU performance claims without measurement

## Rank Layout

```text
rank = dp_rank * (pp_size * tp_size) + pp_rank * tp_size + tp_rank

tp_rank = rank % tp_size
pp_rank = (rank // tp_size) % pp_size
dp_rank = rank // (tp_size * pp_size)
```

Example `dp=1, pp=2, tp=1` (world=2):

```text
rank0: pp0 (embed + layers[0:L/2])
rank1: pp1 (layers[L/2:L] + norm + lm_head)
```

Example `dp=1, pp=2, tp=2` (world=4):

```text
ranks 0,1: pp0 with TP group {0,1}
ranks 2,3: pp1 with TP group {2,3}
```

## Data Flow (one microbatch)

```text
[tokens] → stage0 → hidden ──send──► stage1 → … → stage_{P-1} → loss
                ▲                                      │
                └──────── recv grad ◄──── send grad ───┘
```

## 1F1B Schedule

Let `P = pp_size`, `M = num_microbatches` (`M >= P` recommended).

For stage `p`:

1. **Warmup**: run `(P - 1 - p)` forwards (fill the pipeline).
2. **Steady**: for remaining microbatches, alternate **1 forward + 1 backward**.
3. **Cooldown**: drain remaining backwards.

```text
P=2, M=4, stage0:  F F F F B B B B   (warmup=1 → actually warmup P-1-p=1)
                     more precisely Megatron-style below
```

Megatron-style per-stage warmup count: `num_warmup = P - pp_rank - 1`.

## Correctness Rules

1. First stage consumes `input_ids`; last stage consumes `targets` and computes CE.
2. Non-first stages: received activation must `requires_grad_(True)` before local forward.
3. Sent forward tensors are detached; grads travel explicitly via P2P on backward.
4. Loss is averaged over microbatches on the last stage, then broadcast/reduced for logging.
5. Token accounting: local tokens = `microbatch_size * seq * num_microbatches`;
   global tokens = local × **dp_size** (PP does not multiply tokens).

## Memory vs Bubble Trade-off

| Schedule | Peak activation memory | Bubble |
|----------|------------------------|--------|
| GPipe (all-F then all-B) | O(M) microbatches | smaller compute bubble, huge memory |
| 1F1B | O(P) roughly | some bubble, far less memory |

## Risks

- Deadlock if send/recv order disagrees across ranks.
- Shape mismatch on P2P buffers.
- Forgetting to scale loss by `1/M` → effective LR wrong.
- Wrapping DDP incorrectly across PP ranks (must stay on DP group only).

## Future Work

- DP×PP×TP combined RunPod benchmarks
- Virtual pipeline / interleaved 1F1B
- Overlap P2P with compute
- Distributed checkpoint with PP reshard
