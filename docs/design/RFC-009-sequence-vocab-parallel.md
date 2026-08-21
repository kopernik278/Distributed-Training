# RFC-009: Sequence Parallel and Vocab Parallel

## Motivation

Phase 2–8 shard **weights** (and activations only on the TP hidden axis inside
Column/Row pairs). Activations for LayerNorm / residual / dropout stay
full-sequence `[B, S, H]` on every TP rank — redundant memory and compute.

**Sequence Parallel (SP)** shards the **sequence** axis across the TP group so
LayerNorm and residual run on `[B, S/tp, H]`. Collectives become:

| Layer boundary | Without SP | With SP |
|--|--|--|
| Into ColumnParallel | (identity) | AllGather on seq |
| Out of RowParallel | AllReduce | ReduceScatter on seq |

**Vocab Parallel (VP)** shards the embedding / LM-head vocabulary so logits are
`[B, S, V/tp]` and loss uses a parallel cross-entropy (Megatron-style).

## Design

### Flags

- `--sequence-parallel` → `TrainingConfig.sequence_parallel` (requires `tp_size > 1`,
  `seq_len % tp_size == 0`)
- `--vocab-parallel` → `TrainingConfig.vocab_parallel` (requires `tp_size > 1`,
  `vocab_size % tp_size == 0`)

### Mappings (`mappings.py`)

- `scatter_to_sequence_parallel_region` — split seq (fwd) / AllGather (bwd)
- `gather_from_sequence_parallel_region` — AllGather seq (fwd) / ReduceScatter (bwd)
- `reduce_scatter_to_sequence_parallel_region` — ReduceScatter (fwd) / AllGather (bwd)

### Layers

- `ColumnParallelLinear(sequence_parallel=True)`: AllGather input on seq, then GEMM;
  backward ReduceScatter `dX` instead of AllReduce when SP is on.
- `RowParallelLinear(sequence_parallel=True)`: GEMM then ReduceScatter (not AllReduce).
- `VocabParallelEmbedding`: weight `[V/tp, H]`, mask OOR ids, AllReduce (or
  ReduceScatter when SP) so every rank sees a consistent hidden state.

### Model

- Embed (+ pos) on full sequence → optional scatter into SP before blocks.
- Blocks: LN on SP activations; attention/MLP Column/Row as above; residual in SP.
- Before dense LM head: gather from SP. With VP: `ColumnParallelLinear` LM head
  (`gather_output=False`) tied to vocab embed; `vocab_parallel_cross_entropy`.

### Attention

After ColumnParallel AllGather, QKV sees the **full** sequence (local heads only).
Causal mask stays `[S, S]`. RowParallel ReduceScatter returns SP layout for residual.

## Correctness

- SP off ≡ current TP numerics.
- SP on vs denser/non-SP TP with identical weights: activations/grads match after gather.
- VP CE matches `F.cross_entropy` on the gathered logits (fp32 tests).

## Non-goals

- Sequence-parallel dropout RNG alignment across ranks beyond `torch` defaults
- Fused AG+GEMM kernels / Userbuffers
- Claiming CPU/Gloo tokens/s speedup
