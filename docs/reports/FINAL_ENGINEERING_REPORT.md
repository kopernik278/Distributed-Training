# Final Engineering Report — Mini Distributed Training Engine

**Branch:** `cursor/mini-training-engine-phase1-ed3e`  
**Policy:** Every tok/s figure below comes from RunPod GPU runs written via `--metrics-path`. Profiled JSON is excluded from throughput claims.

---

## 1. Executive summary

This project implements an interview-oriented Megatron/DeepSpeed-style mini LLM engine across **DDP, TP, DP×TP, PP (1F1B), checkpointing, profiler, comm/compute overlap, Sequence Parallel, and Vocab Parallel**, then stress-tests it with:

| Axis | What we ran |
|---|---|
| Model | **Base ~160M** and **XL ~479M** |
| Data | **WikiText-2** and **WikiText-103** (HF parquet → packed LM stream) |
| Hardware | RunPod **2×RTX 4090** (base) and **2×A100 80GB PCIe** (XL) |
| Analysis | Suite summary + pairwise/relative `suite_analysis` |

**Headline (XL / 2×A100 / WikiText-103):** TP≈ matches 1-GPU; **overlap +6.3%**; DDP-2 still regresses on PCIe; SP overhead shrinks vs the small-model suite; **PP completes** on the reduced schedule.

---

## 2. Implementation map (by technique)

| Technique | Role in the engine | Key code / docs |
|---|---|---|
| **DDP** | Replicate model; AllReduce grads; tokens scale with DP | `distributed.py`, RFC-001 |
| **TP** | Shard Column/Row linears; AG/RS/AR on activations | `layers.py`, `mappings.py`, RFC-002 |
| **DP×TP** | 2D groups: `rank = dp*(pp*tp)+…` | `parallel_state.py`, RFC-003 |
| **PP** | Layer split + educational **1F1B** async P2P | `pipeline.py`, RFC-004 |
| **Checkpoint** | Step dirs, TP reshard, resume | `checkpoint.py`, RFC-005 |
| **Profiler** | Named ranges + Chrome traces (`--profile-dir`) | `profiler.py`, RFC-006 |
| **Overlap** | Async ColumnParallel bwd + delayed RowParallel wait | `overlap.py`, RFC-007/008 |
| **SP / VP** | Sequence shard for LN/dropout; vocab-parallel CE | RFC-009, Phase 9 docs |

Harness: `scripts/benchmark_engineering_suite.sh`, `scripts/benchmark_engineering_xl.sh`, `scripts/analyze_engineering_suite.py`.

---

## 3. Workloads

### Suite A — Base (2×4090)

- Model: h=1024, L=12, heads=16, seq=512, batch=2, vocab=8192 → **~160.1M** params (TP shard ~84.5M)
- Data: WikiText-2  
- Commit: `331308d` / PP follow-ups `5f28bd8`  
- Host: NODE topology, no NVLink (PCIe)

Artifacts: [`docs/experiments/engineering_suite/`](../experiments/engineering_suite/)

### Suite B — XL (2×A100 80GB PCIe)

- Model: h=1536, L=16, heads=16, seq=512, batch=1, vocab=16384 → **~479.3M** params  
- Data: **WikiText-103** (Salesforce HF parquet shards converted to `wiki.train.raw`)  
- Commit: `a9879e3`  
- Host: PHB topology, PCIe Gen4×16

Artifacts: [`docs/experiments/engineering_suite_xl/`](../experiments/engineering_suite_xl/)

---

## 4. Measured results

### 4.1 Suite A — 2×4090 / WikiText-2 / ~160M

| case | dp/tp/pp | tok/s | vs 1-GPU | bwd share of step | note |
|---|---|---:|---:|---:|---|
| gpu1_ddp | 1/1/1 | **18289** | 1.00× | 47% | baseline |
| gpu2_ddp | 2/1/1 | **9535** | 0.52× | **87%** | AllReduce-bound |
| gpu2_tp | 1/2/1 | **11446** | 0.63× | 48% | best 2-GPU eng |
| gpu2_tp_overlap | 1/2/1 | **11342** | 0.62× | 48% | ≈ flat (−0.9%) |
| gpu2_tp_sp | 1/2/1 | **9337** | 0.51× | 49% | SP −18% vs TP |
| gpu2_tp_sp_vp | 1/2/1 | **9262** | 0.51× | 49% | VP ≈ neutral |

PP eng-seq hung on this NODE/PCIe box; mid/reduced PP JSON recorded separately (`gpu2_pp_mid`, `gpu2_pp_reduced`).

Prior Phase-8 overlap reference (same class of 4090s, different schedule): **+4.2%** at batch=4 / h=1024; **−2.9%** at h=2048 batch=2 — see `docs/experiments/phase8-runpod-overlap-results.md`.

### 4.2 Suite B — 2×A100 / WikiText-103 / ~479M

| case | dp/tp/pp | tok/s | vs 1-GPU | bwd share | note |
|---|---|---:|---:|---:|---|
| gpu1_ddp | 1/1/1 | **3837** | 1.00× | 53% | XL single-GPU |
| gpu2_ddp | 2/1/1 | **2029** | 0.53× | **88%** | still AllReduce-bound |
| gpu2_tp | 1/2/1 | **4051** | **1.06×** | 52% | TP beats 1-GPU |
| gpu2_tp_overlap | 1/2/1 | **4306** | **1.12×** | 48% | **+6.3% vs sync TP** |
| gpu2_tp_sp | 1/2/1 | **3786** | 0.99× | 52% | SP −6.5% vs TP |
| gpu2_tp_sp_vp | 1/2/1 | **3859** | 1.01× | 52% | VP +1.9% on SP |
| gpu2_pp | 1/1/2 | **5493** | 1.43×* | n/a† | reduced seq/mb schedule |

\*PP uses `seq=256`, `batch=1`, `num_microbatches=2` (suite default for stability); not identical to TP schedule.  
†PP timer currently folds schedule into forward (`bwd_ms=0`); use step time / tok/s.

---

## 5. Analysis by technique

### 5.1 DDP

On **both** measured fabrics (4090 NODE and A100 PHB), 1→2 DDP **halves** global tok/s while backward becomes ~87–88% of the step. Gradient AllReduce volume dominates when interconnect is PCIe and the model is not huge enough to amortize it.

**Interview takeaway:** DDP scaling is an interconnect story first; weak multi-GPU PCIe hosts are excellent demos of *negative* scaling.

### 5.2 Tensor Parallel

TP shards parameters (~½ memory) and replaces one large grad AllReduce with finer activation collectives. On 4090s TP beats DDP-2; on A100 XL, TP **slightly exceeds** 1-GPU tok/s (1.06×) because memory traffic / optimizer cost per rank drops.

### 5.3 Communication overlap

Overlap is **workload-dependent**:

- Base eng (batch=2, ~160M, 4090): **−0.9%** (noise / not enough GEMM to hide NCCL)
- Phase-8 batch=4: **+4.2%**
- XL (~479M, A100): **+6.3%** with identical loss vs sync TP

Larger compute intensity makes delayed waits and async ColumnParallel bwd pay off.

### 5.4 Sequence / Vocab Parallel

SP adds AG/RS around non-TP regions. Overhead fell from **−18%** (base/4090) to **−6.5%** (XL/A100) — more compute hides collectives. VP on top is roughly neutral (+1.9% here) while cutting vocab-layer memory.

### 5.5 Pipeline Parallel

1F1B with async P2P is functionally correct (unit tests + GPU runs). Eng-default `seq=512/batch=2/mb=4` stalled on weak 4090 NODE links; **A100 suite `gpu2_pp` completed** with the reduced schedule. PP tok/s is not apples-to-apples with TP rows (different microbatching).

### 5.6 Profiler

`--profile-dir` runs inflate step time (XL profiled ~913 ms vs ~126 ms). Use traces qualitatively (`tp_all_reduce`, matmuls); never publish profiled tok/s.

### 5.7 Data & model scale

- **WikiText-2**: small, fast, good for CI-like GPU loops.  
- **WikiText-103**: ~300MB parquet → large packed stream; exercises real vocab skew / longer epochs.  
- **XL ~479M**: enough to show overlap wins and keep TP memory comfortable on 80GB with batch=1.

---

## 6. Larger distributed systems (GPU count / multi-node)

| Attempt | Outcome |
|---|---|
| 4×4090 (earlier) | Pod up; `nvidia-smi` OK; **PyTorch CUDA init failed** (driver/CUDA mismatch). No numbers claimed. |
| 4×4090 (retry) | **No stock** |
| 2×4090 (XL retry) | Stuck without published SSH port 22 → deleted |
| 2×A100 | **Success** — full XL matrix |
| Multi-node Instant Cluster | Launcher exists (`scripts/runpod/run_ddp_multinode.sh`); **not stocked / not executed** this pass |

We do **not** invent multi-node or 4-GPU tok/s. The multinode script is ready when a cluster provides `MASTER_ADDR` / `NUM_NODES` / `NODE_RANK`.

---

## 7. Comparative takeaways

1. **Prefer TP over DP** on PCIe-only multi-GPU boxes for this model class.  
2. **Enable overlap** when GEMM is large enough (XL showed +6.3%; tiny batches may regress).  
3. **SP/VP** are primarily memory/correctness features; throughput cost shrinks as the model grows.  
4. **PP** needs adequate P2P bandwidth or reduced microbatch shapes on consumer fabrics.  
5. **Profiler** is for communication/compute timelines, not for published throughput.

---

## 8. Reproduce

```bash
# Base suite (WikiText-2, ~160M)
./scripts/runpod/bootstrap.sh
GPU_COUNT=2 PROFILE=0 ./scripts/benchmark_engineering_suite.sh

# XL suite (WikiText-103, ~479M)
SCALE=xl DATASET=wikitext103 GPU_COUNT=2 PROFILE=1 \
  ./scripts/benchmark_engineering_xl.sh
```

Analysis: `python3 scripts/analyze_engineering_suite.py <out_dir>`

---

## 9. Limitations

1. Single-node measurements only (2 GPUs).  
2. No NVLink on measured hosts.  
3. 4-GPU / multi-node not successfully measured (stock or CUDA/SSH blockers).  
4. PP suite shape differs from TP rows by design (stability).  
5. WikiText-103 packing uses a char budget (~80M) so the id tensor fits in RAM — still far larger than WikiText-2.

---

## 10. Conclusion

The engine covers the parallel-training technique stack expected in AI Infra interviews, validated on **real RunPod GPUs** with **two model scales** and **two real corpora**. The data show when each knob helps (TP + overlap on XL/A100) and when the fabric dominates (DDP on PCIe). Artifacts and pairwise analysis live under `docs/experiments/engineering_suite*` for audit.
