# v38 Fig. 4/5 Diagnostic Data

Suite: EPFL20; vector seed: `7089`; vectors: `128`; budgets: 5%, 10%, and 20%.

Fig. 4 is a rank-profile diagnostic against the random-vector oracle ranking. It is not the primary metric.
Fig. 5 is a circuit-budget gain distribution derived from Table 4 component-ablation rows.

Important: SEGR uses `global_rank` / `chosen_family` from `node_debug.csv`; `final_score` is an intermediate residual score and is not re-sorted here.

## Fig. 4 Taylor Summary

| Method | Corr. | Centered RMSE | Norm. CRMSE | Std. ratio |
|---|---:|---:|---:|---:|
| SEGR | 0.5421 | 0.2695 | 0.9314 | 1.0000 |
| Static-Prox | 0.4962 | 0.2757 | 0.9537 | 1.0000 |
| SEGR w/o GNN | 0.3132 | 0.3335 | 1.1530 | 1.0000 |
| GNN-only | 0.1701 | 0.3708 | 1.2819 | 1.0000 |
| Shuffled-GNN | 0.0121 | 0.4060 | 1.4038 | 1.0000 |
| Cache-Struct | 0.1085 | 0.3807 | 1.3168 | 1.0000 |
| Struct-Family | 0.3139 | 0.3329 | 1.1514 | 1.0000 |

## Fig. 5 Gain Distribution Summary

| Variant | Positive | Zero | Negative | Mean delta | Median delta | Q25 | Q75 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Static-Prox | 44 | 16 | 0 | 0.1507 | 0.0687 | 0.0000 | 0.2583 |
| SEGR w/o GNN | 49 | 8 | 3 | 0.2019 | 0.1675 | 0.0248 | 0.3282 |
| GNN-only | 57 | 3 | 0 | 0.3565 | 0.3289 | 0.2021 | 0.4763 |
| Shuffled-GNN | 57 | 3 | 0 | 0.4819 | 0.4583 | 0.3147 | 0.7192 |
| Cache-Struct | 55 | 5 | 0 | 0.3933 | 0.3755 | 0.1765 | 0.5975 |
| Struct-Family | 49 | 9 | 2 | 0.2554 | 0.2712 | 0.0366 | 0.4255 |
