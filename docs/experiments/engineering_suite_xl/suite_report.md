# Engineering suite summary

Cases: **9**

| case | dp/tp/pp | overlap | SP/VP | step ms | tok/s | loss |
|---|---|---|---|---:|---:|---:|
| gpu1_ddp | 1/1/1 | False | 0/0 | 133.43 | 3837 | 7.3634 |
| gpu2_ddp | 2/1/1 | False | 0/0 | 504.79 | 2029 | 7.2294 |
| gpu2_pp | 1/1/2 | False | 0/0 | 93.21 | 5493 | 7.5977 |
| gpu2_tp | 1/2/1 | False | 0/0 | 126.38 | 4051 | 7.1260 |
| gpu2_tp_overlap | 1/2/1 | True | 0/0 | 118.92 | 4306 | 7.1260 |
| gpu2_tp_overlap_profiled | 1/2/1 | True | 0/0 | 1253.71 | 3401 | 7.7056 |
| gpu2_tp_profiled | 1/2/1 | False | 0/0 | 912.89 | 3210 | 7.7056 |
| gpu2_tp_sp | 1/2/1 | False | 1/0 | 135.22 | 3786 | 7.1914 |
| gpu2_tp_sp_vp | 1/2/1 | False | 1/1 | 132.67 | 3859 | 7.4414 |

Raw: `suite_summary.json`
