# RFC-005: Distributed Checkpoint (Save / Resume / Reshard)

## Motivation

After DP / TP / PP, a training engine must **survive restarts** and **change
parallel topology** without retraining from scratch.

Interview-relevant skills:

1. Save only unique model shards (skip redundant DP replicas).
2. Resume with the **same** `(dp, pp, tp)` layout.
3. **Reshard** tensor-parallel weights when `tp_size` changes (gather → split).

## Goals

1. Sharded checkpoint directory per training step.
2. Metadata describing config + parallel topology + format version.
3. Same-layout resume of model (+ optimizer).
4. TP reshard for `MiniTransformerLM` / `PipelineStage` local modules (`tp=1↔tp=2`).
5. CLI hooks: `--checkpoint-dir`, `--save-interval`, `--resume`.
6. Unit tests for round-trip and TP reshard numerical equality.

## Non-goals

- Full PyTorch Distributed Checkpoint (DCP) / TorchTitan integration
- Optimizer-state reshard across TP (model weights only on topology change)
- PP topology change (`pp=1↔pp=2`) in this milestone
- Async / non-blocking checkpoint I/O
- ZeRO optimizer-state partitioning

## Directory Layout

```text
checkpoints/step_000020/
  metadata.json
  mp_rank_pp000_tp000.pt
  mp_rank_pp000_tp001.pt   # when tp=2
  ...
```

Each `.pt` file (written by `dp_rank == 0` only):

```python
{
  "format_version": 1,
  "step": 20,
  "pp_rank": 0,
  "tp_rank": 0,
  "tp_size": 2,
  "pp_size": 1,
  "model": state_dict,
  "optimizer": state_dict | None,
  "param_specs": { name: {"tensor_model_parallel": bool, "partition_dim": int|None} },
}
```

`metadata.json` (written once by global rank 0):

```json
{
  "format_version": 1,
  "step": 20,
  "config": { ... },
  "parallel": {"data_parallel_size": 1, "pipeline_parallel_size": 1, "tensor_parallel_size": 2}
}
```

## Save Rules

1. Unwrap DDP before `state_dict()`.
2. Only `dp_rank == 0` writes the shard for `(pp_rank, tp_rank)`.
3. Barrier after all writers finish.
4. Tag ColumnParallel params with `partition_dim=0`, RowParallel weights with `partition_dim=1`.

## Resume (same topology)

1. Read `metadata.json`; assert `pp_size`/`tp_size` match current world.
2. Load `mp_rank_ppXXX_tpYYY.pt` for this rank's `(pp, tp)`.
3. `load_state_dict` into local module; optionally restore optimizer.

## TP Reshard

```text
source shards (tp_src) --consolidate along partition_dim--> full tensor
full tensor --split for tp_dst / tp_rank--> local shard
```

- Replicated tensors (embeddings, LayerNorm, Row bias, …): copy from shard 0.
- Column weight/bias: `torch.cat(..., dim=0)` then slice.
- Row weight: `torch.cat(..., dim=1)` then slice.

On topology change, **optimizer state is not restored** (re-init); document this clearly.

## Correctness Tests

1. Save → load same `tp=1`: parameter equality.
2. Save → load same `tp=2`: shard equality.
3. Train briefly, save `tp=1`, reshard-load into `tp=2`, compare forward logits to dense reference built from consolidated weights.
4. Smoke: `--resume` continues with finite loss.

## Risks

- Missing `partition_dim` metadata → silent wrong concat axis.
- DP ranks all writing → torn files / races.
- Loading tied `lm_head`/`embed` across PP stages incorrectly.

## Future Work

- PP reshard / consolidate full pipeline into one LM
- Adam state reshard
- Async checkpoint + checksums
- HuggingFace / Safetensors export
