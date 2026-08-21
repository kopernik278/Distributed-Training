# Engineering suite summary

Cases: **10**

| case | dp/tp/pp | overlap | SP/VP | step ms | tok/s | loss |
|---|---|---|---|---:|---:|---:|
| gpu1_ddp | 1/1/1 | False | 0/0 | 56.08 | 18289 | 6.6205 |
| gpu2_ddp | 2/1/1 | False | 0/0 | 215.36 | 9535 | 6.6642 |
| gpu2_pp_mid | 1/1/2 | False | 0/0 | 68.21 | 30027 | 6.8547 |
| gpu2_pp_reduced | 1/1/2 | False | 0/0 | 51.23 | 9997 | 6.6826 |
| gpu2_tp | 1/2/1 | False | 0/0 | 89.50 | 11446 | 6.6588 |
| gpu2_tp_overlap | 1/2/1 | True | 0/0 | 90.28 | 11342 | 6.6588 |
| gpu2_tp_overlap_profiled | 1/2/1 | True | 0/0 | 661.22 | 9002 | 7.3746 |
| gpu2_tp_profiled | 1/2/1 | False | 0/0 | 616.41 | 9091 | 7.3746 |
| gpu2_tp_sp | 1/2/1 | False | 1/0 | 109.67 | 9337 | 6.6881 |
| gpu2_tp_sp_vp | 1/2/1 | False | 1/1 | 110.57 | 9262 | 6.6532 |

Raw: `suite_summary.json`

