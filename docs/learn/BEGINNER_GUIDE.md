# Learning Guide: Mini Distributed Training Engine

> Audience: you are learning AI Infra, not already a senior training-systems engineer.
> Read this **before** diving into RFC files. Code citations use repo paths.

This project is a **tiny Megatron-style training engine**. The model is small on purpose.
The point is to *see* DDP, Tensor Parallel, Pipeline Parallel, checkpointing, and profiling
without 70B-parameter noise.

---

## 0. The one picture you need

Training one optimizer step is always:

```text
data → forward (compute logits) → loss → backward (compute grads) → optimizer
```

**Distributed training only changes *where* tensors live and *when* GPUs talk.**

| Phase | What we split | How GPUs talk | Analogy |
|--|--|--|--|
| 1 DDP | **data** (same full model on each GPU) | AllReduce **gradients** after backward | Several students copy the same textbook, each reads different pages, then they average their notes |
| 2 TP | **matrix columns/rows inside a layer** | AllReduce / AllGather **activations** inside the layer | One huge spreadsheet split by columns; you pass partial sums to finish the product |
| 3 DP×TP | data *and* matrices | DDP group + TP group (two radios) | Textbook copies *of already-split* spreadsheets |
| 4 PP | **layers** (depth) | Point-to-point send/recv of activations | Assembly line: GPU0 does layers 1–2, GPU1 does 3–4 |
| 5 Ckpt | — | Disk: unique shards only | Save puzzle pieces; later reassemble with a different number of pieces (TP reshard) |
| 6 Prof | — | — | Stopwatch that labels “compute” vs “waiting for network” |

If you remember only one sentence: **DDP communicates after the layer; TP communicates inside the layer; PP communicates between stages.**

---

## 1. Words you will keep seeing

- **Rank**: one process, usually one GPU. Rank 0 is the “logger”.
- **World size**: how many ranks.
- **Process group**: a *subset* of ranks that are allowed to call `all_reduce` together. We have TP groups, DP groups, PP groups.
- **Collective**: everyone in a group participates (`all_reduce`, `all_gather`, `broadcast`) or it **deadlocks**.
- **P2P**: only two ranks (`send`/`recv`). Pipeline uses this.
- **Shard**: a slice of a tensor. TP stores `1/tp` of a weight.
- **Microbatch**: a small batch used to fill a pipeline. Several microbatches = one optimizer step.
- **Gloo vs NCCL**: Gloo = CPU/network library (correctness on laptops). NCCL = GPU collective library (real speed). **Never quote Gloo tokens/s as GPU performance.**

---

## 2. Phase 1 — DDP (the baseline)

**Problem:** one GPU is too slow. Put the *same* model on 2 GPUs, feed *different* batches, average gradients so both copies stay identical.

**What the code does:** `DistributedDataParallel` wraps the model. During `loss.backward()`, PyTorch AllReduces each parameter’s gradient.

**What you should be able to explain in an interview:**

1. After one step, `model.weight` on rank0 equals rank1 (same init + same averaged grads + same optimizer).
2. Global tokens/s ≈ local tokens × **dp_size** (each replica saw different tokens).
3. Scaling is rarely 2.0×: backward gets longer because AllReduce is inside it.

**Try:** `NPROC_PER_NODE=2 ./scripts/run_single_node_ddp.sh --steps 5`

---

## 3. Phase 2 — Tensor Parallel (the hard idea)

Dense MLP: `y = W2 · gelu(W1 · x)`.

Megatron split:

```text
W1 split by **output** columns  → ColumnParallelLinear  (each GPU computes part of gelu input)
W2 split by **input** rows      → RowParallelLinear     (each GPU computes a partial y, then SUM)
```

**The scary bit is autograd**, not the math. Communication in forward has a *partner* in backward:

| Name in `mappings.py` | Forward | Backward |
|--|--|--|
| copy_to_TP | do nothing | AllReduce grads of the shared input |
| reduce_from_TP | AllReduce | do nothing |
| gather | AllGather | split |
| scatter | split | AllGather |

If you only AllReduce in forward and forget backward, the model **silently trains wrong**. That is why we have `tests/test_tensor_parallel.py` comparing sharded MLP to `nn.Linear`.

**Token counting trap:** every TP rank sees the *same* batch. Do **not** multiply tokens by `tp_size`.

**Try:** `./scripts/run_tp.sh 2 --steps 5 --hidden-size 64 --num-heads 8`

---

## 4. Phase 3 — DP × TP (two radios)

`world = dp × tp`. Example 4 GPUs, tp=2:

```text
ranks 0,1 = replica A (TP pair, same batch A)
ranks 2,3 = replica B (TP pair, same batch B)
DDP AllReduce only between 0↔2 and 1↔3 (matching shards)
```

**Seeding rules (easy to get wrong):**

- Init seed uses **tp_rank** so matching shards start the same, then **broadcast inside DP**
- Data seed uses **dp_rank** so TP peers share a batch, DP replicas differ

---

## 5. Phase 4 — Pipeline + 1F1B

Split *layers*, not matrices. Stage0: embed + first half of blocks. Last stage: rest + LM head + loss.

**1F1B** = after filling the pipe, do one forward and one backward so you do not store *all* microbatch activations (GPipe would).

**Deadlock:** in steady state stage0 wants to send the next activation while stage1 wants to send the previous gradient. Blocking `send`+`send` waits forever. We use `isend`/`irecv`.

**Numeric check:** PP loss/grads match an *untied* dense model. (Tied embed↔lm_head cannot live on two stages.)

---

## 6. Phase 5 — Checkpoint

Save **unique** pieces (`dp_rank==0` only). File name encodes `pp` and `tp`.

**Reshard `tp=1 → tp=2`:** concatenate along `partition_dim`, then slice. Optimizer Adam slots depend on tensor *shapes*, so we **drop optimizer state** when TP changes.

---

## 7. Phase 6–8 — Profiling then overlap

Profiler = labeled stopwatch (Chrome trace). Overlap = hide AllReduce behind compute
that does not need the AllReduce result yet.

**ColumnParallel backward** (Phase 7):

```text
dX_local = dY @ W     → start AllReduce(dX)
dW = X^T @ dY         → can run at the same time
then wait for AllReduce
```

**RowParallel forward** (Phase 8):

```text
local GEMM → start AllReduce(Y)
  (caller may do other work)
flush → add bias → residual → next layer
```

`TransformerBlock` uses Megatron-style `skip_bias_add` and
`finalize_tensor_parallel_output` so the wait sits at the residual, not inside
the collective helper.

Turn on with `--overlap`. Tiny CPU models will not get faster; this is a GPU pattern.

---

## 8. How you can participate (practical)

You do not need to write kernels. Useful loops:

1. **Read one file** (`mappings.py` or `pipeline.py`) and restate it in your own words.
2. **Run one command** on CPU (Gloo) and paste the JSON `summary` back in chat — we interpret it together.
3. **Ask “why this collective?”** for any function you do not understand.
4. **Change one knob** (`tp_size`, `num_microbatches`) and predict tokens/s *direction* before running.

Suggested first homework:

```bash
python3 -m unittest tests.test_tensor_parallel tests.test_checkpoint -v
BACKEND=gloo ./scripts/run_tp.sh 1 --steps 3 --batch-size 2 --seq-len 32 --hidden-size 32 --num-heads 4 --dropout 0.0 --profile-dir /tmp/prof_demo
```

Then open `/tmp/prof_demo/rank0_summary.txt` (Chrome JSON on CPU is still useful for CPU ops).

---

## 9. Map of source files

| File | Job |
|--|--|
| `parallel_state.py` | Who is in which group (dp/pp/tp ranks) |
| `mappings.py` | TP collectives + backward |
| `layers.py` | Column/Row Linear |
| `model.py` | Transformer + `PipelineStage` |
| `pipeline.py` | 1F1B schedule |
| `checkpoint.py` | save / resume / TP reshard |
| `profiler.py` | Chrome trace export |
| `train.py` | CLI glue |
