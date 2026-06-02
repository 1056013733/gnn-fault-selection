# Paper-Facing Table 3/4 Summary

Source package: `v38_random_vector_package_repro_20260526_01_work`

Protocol: EPFL20, seed 7089, 128 random vectors, budgets 5%, 10%, and 20%. All values are macro averages over the 20 held-out circuits unless noted otherwise.

Important ranking field: SEGR and architecture variants must be evaluated from `global_rank` / `chosen_family` in `node_debug.csv`, not by re-sorting `final_score`.

## Table 3: Same-Feature LOCO Supervised Baselines

| Method | Learning type | Training labels | Macro ideal ratio | Loss rows | Random-gate failures | >=0.75 no-loss circuits |
|---|---|---|---:|---:|---:|---:|
| SEGR | label-free ranking | none | 0.8185 | 0 | 0 | 13 |
| LR-LOCO | supervised LOCO | 19 training circuits | 0.7052 | 20 | 20 | 3 |
| RF-LOCO | supervised LOCO | 19 training circuits | 0.6588 | 29 | 20 | 2 |
| MLP-LOCO | supervised LOCO | 19 training circuits | 0.6557 | 31 | 20 | 3 |
| FuSa-LOCO | supervised LOCO | 19 training circuits | 0.6557 | 31 | 20 | 3 |

## Table 4(a): Component Ablations

| Variant | Changed component | Macro ideal ratio | Loss rows | Random-gate failures | >=0.75 no-loss circuits |
|---|---|---:|---:|---:|---:|
| SEGR | full method | 0.8185 | 0 | 0 | 13 |
| Static-Prox | static proximity only | 0.6678 | 0 | 8 | 9 |
| SEGR w/o GNN | removes GNN rank/final score | 0.6166 | 29 | 7 | 3 |
| GNN-only | GNN rank only | 0.4620 | 36 | 14 | 3 |
| Shuffled-GNN | circuit-local shuffled GNN rank | 0.3366 | 48 | 17 | 2 |
| Cache-Struct | cache structural rank only | 0.4253 | 42 | 14 | 3 |
| Struct-Family | fixed structural family heuristic | 0.5631 | 30 | 8 | 3 |

## Table 4(b): Architecture Sensitivity

| Variant | Hidden | Layers | Macro ideal ratio | Loss rows | Random-gate failures | >=0.75 no-loss circuits |
|---|---:|---:|---:|---:|---:|---:|
| SEGR | 64 | 2 | 0.8185 | 0 | 0 | 13 |
| SEGR-H32 | 32 | 2 | 0.7561 | 4 | 2 | 11 |
| SEGR-H128 | 128 | 2 | 0.7888 | 1 | 2 | 12 |
| SEGR-L1 | 64 | 1 | 0.8063 | 0 | 1 | 13 |
| SEGR-L3 | 64 | 3 | 0.7658 | 3 | 2 | 11 |

