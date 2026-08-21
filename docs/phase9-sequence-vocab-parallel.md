# Phase 9: Sequence Parallel and Vocab Parallel

## What landed

- RFC: `docs/design/RFC-009-sequence-vocab-parallel.md`
- Sequence mappings: scatter / gather / reduce-scatter on the **sequence** dim
- `ColumnParallelLinear(sequence_parallel=True)`: AllGather → GEMM; bwd ReduceScatter(`dX`)
- `RowParallelLinear(sequence_parallel=True)`: GEMM → ReduceScatter (not AllReduce)
- `VocabParallelEmbedding` + `vocab_parallel_cross_entropy`
- CLI: `--sequence-parallel`, `--vocab-parallel`
- Tests: `tests/test_sequence_vocab_parallel.py`

## Mental model

```text
Without SP (TP only):
  LN/residual on full [B,S,H] everywhere
  RowParallel ends with AllReduce

With SP:
  after embed: scatter seq → [B, S/tp, H]
  LN/residual local on SP activations
  Column: AllGather seq → GEMM (attention sees full S, local heads)
  Row: GEMM → ReduceScatter seq
```

Vocab parallel shards `V` on embed + LM head; loss never materializes full `[B,S,V]`.

## How to run

```bash
# TP=2 + sequence parallel (seq_len must be divisible by tp)
BACKEND=gloo ./scripts/run_tp.sh 2 --steps 3 --seq-len 64 --hidden-size 64 \
  --num-heads 8 --sequence-parallel

# + vocab parallel (vocab_size % tp == 0; default vocab 4096 is fine)
BACKEND=gloo ./scripts/run_tp.sh 2 --steps 3 --seq-len 64 --hidden-size 64 \
  --num-heads 8 --sequence-parallel --vocab-parallel

python3 -m unittest tests.test_sequence_vocab_parallel -v
```

## Next

- Optional Adam-state TP/VP reshard polish
- Optional RunPod memory comparison (SP activation footprint)
