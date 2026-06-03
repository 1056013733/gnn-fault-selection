# Supplemental SCOAP and Selector Diagnostics

These supplemental rows do not change or retune SEGR. Ranking orders are fixed before offline FI counts, RV-Oracle counts, random-vector outcomes, held-out labels, baseline outcomes, or evaluation metrics are opened.

The selector-family sweep is diagnostic only: it evaluates fixed runtime-visible candidate rankings after the rankings have been emitted, and it must not be used to select or retune SEGR.

## Method Summary

| Method | Circuits | Macro ideal | Loss rows | Random gate fails | >=0.75 no-loss circuits |
| --- | ---: | ---: | ---: | ---: | ---: |
| scoap_avg_fault_cost | 20 | 0.7595 | 16 | 3 | 6 |
| scoap_co_only | 20 | 0.7837 | 12 | 2 | 10 |
| scoap_min_fault_cost | 20 | 0.7744 | 15 | 3 | 8 |
| scoap_worst_fault_cost | 20 | 0.6973 | 25 | 3 | 4 |
| segr_structure_derived_selector | 20 | 0.8144 | 0 | 0 | 13 |
| standard_scoap_testability | 20 | 0.7744 | 15 | 3 | 8 |

## Selector Family Diagnostics

| Circuit | Chosen family | Rank among families | Chosen ratio | Best family | Best ratio | Median family ratio |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| adder | mass_balance | 2 | 1.0000 | eigen | 1.0000 | 0.8588 |
| arbiter | pdom_dist | 5 | 0.9267 | union_dist_pr_topo_edge_pr | 0.9886 | 0.4474 |
| bar | pdom_dist | 5 | 0.9814 | final_score | 0.9814 | 0.6950 |
| cavlc | pdom_dist | 1 | 0.9008 | pdom_dist | 0.9008 | 0.4475 |
| ctrl | pdom_dist | 6 | 0.9519 | inv_depth | 0.9642 | 0.7302 |
| dec | index_mid | 12 | 1.0000 | static_proximity | 1.0000 | 1.0000 |
| div | inv_depth | 1 | 0.7599 | inv_depth | 0.7599 | 0.3000 |
| hyp | index_mid | 1 | 0.8035 | index_mid | 0.8035 | 0.3644 |
| i2c | pdom_dist | 5 | 0.9883 | static_proximity | 0.9883 | 0.4734 |
| int2float | pdom_dist | 1 | 0.9212 | pdom_dist | 0.9212 | 0.5853 |
| log2 | sink_reach_near_pr | 1 | 0.6610 | sink_reach_near_pr | 0.6610 | 0.4427 |
| max | pdom_dist | 1 | 0.5036 | pdom_dist | 0.5036 | 0.3679 |
| mem_ctrl | pdom_dist | 1 | 0.9122 | pdom_dist | 0.9122 | 0.2622 |
| multiplier | pdom_dist | 6 | 0.5928 | sink_reach_near_pr | 0.7346 | 0.4571 |
| priority | wl3_common_dist | 4 | 0.7110 | mass_balance | 0.7508 | 0.6524 |
| router | pdom_dist | 1 | 0.7835 | pdom_dist | 0.7835 | 0.2873 |
| sin | mass_balance | 2 | 0.7462 | cache_struct | 0.7534 | 0.5748 |
| sqrt | role_depth_resonance_idx_inv_dist | 2 | 0.6914 | role_depth_resonance_orig_idx_inv_dist | 0.6914 | 0.3835 |
| square | mass_balance | 2 | 0.8383 | out_deg | 0.8395 | 0.7959 |
| voter | index_mid | 2 | 0.6138 | inv_depth | 0.6144 | 0.3886 |
