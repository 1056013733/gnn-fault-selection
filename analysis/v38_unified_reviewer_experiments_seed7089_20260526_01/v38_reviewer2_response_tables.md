# SEGR Reviewer 2 Response Tables

## Method Summary

| Experiment | Method | Macro ideal | Loss rows | Random gate fails |
| --- | --- | ---: | ---: | ---: |
| main | segr_structure_derived_selector | 0.8144 | 0 | 0 |
| same_feature | logistic_regression_same_feature_loco | 0.6962 | 17 | 20 |
| same_feature | mlp_same_feature_loco | 0.6345 | 33 | 20 |
| same_feature | random_forest_same_feature_loco | 0.7039 | 25 | 20 |
| architecture | segr_h128 | 0.7888 | 1 | 2 |
| architecture | segr_h32 | 0.7561 | 4 | 2 |
| architecture | segr_l1 | 0.8063 | 0 | 1 |
| architecture | segr_l3 | 0.7658 | 3 | 2 |
| architecture | segr_structure_derived_selector | 0.8144 | 0 | 0 |
| component | cache_structural_only | 0.4750 | 38 | 13 |
| component | gnn_only | 0.4607 | 36 | 14 |
| component | no_gnn | 0.6914 | 18 | 4 |
| component | pure_static_proximity | 0.6678 | 0 | 8 |
| component | segr_structure_derived_selector | 0.8144 | 0 | 0 |
| component | shuffled_gnn | 0.3364 | 48 | 17 |
| component | structural_family_only | 0.5631 | 30 | 8 |
| fusa | fusa_supervised_fair_mlp_loco | 0.6345 | 33 | 20 |
