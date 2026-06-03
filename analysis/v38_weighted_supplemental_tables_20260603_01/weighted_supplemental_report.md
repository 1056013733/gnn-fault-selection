# Weighted Supplemental Evidence Tables

All weighted ideal ratios use EPFL20 eligible-node counts from the materialized seed-7089 main run.
RV-count sensitivity reports complete EPFL20 rows for 32, 64, 128, and 256 vectors.

## RV-count Sensitivity

| vectors | status | circuits | macro_ideal_ratio | node_weighted_ideal_ratio | loss_rows | random_failures |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | complete | 20 | 0.8027 | 0.7735 | 1 | 0 |
| 64 | complete | 20 | 0.8141 | 0.7887 | 0 | 0 |
| 128 | complete | 20 | 0.8144 | 0.7797 | 0 | 0 |
| 256 | complete | 20 | 0.8170 | 0.7871 | 0 | 0 |

## Runtime / Scaling Cost

| component | sum_seconds | mean_seconds_per_circuit | node_weighted_mean_seconds | max_seconds |
| --- | --- | --- | --- | --- |
| feature_seconds | 14.5228 | 0.7261 | 4.1127 | 7.7386 |
| gnn_seconds | 12.1376 | 0.6069 | 1.5880 | 2.8639 |
| selector_seconds | 90.7809 | 4.5390 | 26.2185 | 49.6793 |
| total_seconds | 134.5308 | 6.7265 | 36.8252 | 69.4728 |

## Absolute-count / Oracle-count Weighted

| method | budget | H_method | H_static | O | oracle_count_weighted_ideal_ratio |
| --- | --- | --- | --- | --- | --- |
| segr_structure_derived_selector | 0.05 | 1954938.0000 | 1380042.0000 | 2412643.0000 | 0.8103 |
| segr_structure_derived_selector | 0.1 | 3537609.0000 | 2073823.0000 | 4414693.0000 | 0.8013 |
| segr_structure_derived_selector | 0.2 | 5639681.0000 | 3045338.0000 | 7685130.0000 | 0.7338 |
