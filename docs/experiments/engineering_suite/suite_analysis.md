# Engineering suite analysis

Baseline: `gpu1_ddp` @ 18289 tok/s

## Relative throughput

| case | dp/tp/pp | tok/s | vs baseline | fwd% | bwd% | opt% |
|---|---|---:|---:|---:|---:|---:|
| gpu1_ddp | 1/1/1 | 18289 | 1.00× | 23% | 47% | 28% |
| gpu2_ddp | 2/1/1 | 9535 | 0.52× | 6% | 87% | 7% |
| gpu2_pp_mid | 1/1/2 | 30027 | 1.64× | 95% | 0% | 3% |
| gpu2_pp_reduced | 1/1/2 | 9997 | 0.55× | 83% | 0% | 16% |
| gpu2_tp | 1/2/1 | 11446 | 0.63× | 42% | 48% | 10% |
| gpu2_tp_overlap | 1/2/1 | 11342 | 0.62× | 42% | 48% | 9% |
| gpu2_tp_sp | 1/2/1 | 9337 | 0.51× | 43% | 49% | 8% |
| gpu2_tp_sp_vp | 1/2/1 | 9262 | 0.51× | 43% | 49% | 7% |

## Pairwise deltas

| comparison | A tok/s | B tok/s | B/A | Δ% |
|---|---:|---:|---:|---:|
| DDP scale 1→2 | 18289 | 9535 | 0.521× | -47.9% |
| DDP-2 vs TP-2 | 9535 | 11446 | 1.200× | +20.0% |
| TP overlap vs sync | 11446 | 11342 | 0.991× | -0.9% |
| Sequence Parallel overhead | 11446 | 9337 | 0.816× | -18.4% |
| Vocab Parallel on top of SP | 9337 | 9262 | 0.992× | -0.8% |

Raw: `suite_analysis.json`
