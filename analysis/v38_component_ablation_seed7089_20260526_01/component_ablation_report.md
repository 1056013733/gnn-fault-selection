# SEGR Component / Method Ablation

All component variants use one global ranking per circuit and are evaluated under the selected seed7089 random-vector oracle.

| Method | Circuits | Macro ideal | Loss rows | Random gate fails | >=0.75 no-loss circuits |
| --- | ---: | ---: | ---: | ---: | ---: |
| cache_structural_only | 20 | 0.4750 | 38 | 13 | 4 |
| gnn_only | 20 | 0.4607 | 36 | 14 | 3 |
| no_gnn | 20 | 0.6914 | 18 | 4 | 5 |
| pure_static_proximity | 20 | 0.6678 | 0 | 8 | 9 |
| segr_structure_derived_selector | 20 | 0.8144 | 0 | 0 | 13 |
| shuffled_gnn | 20 | 0.3364 | 48 | 17 | 2 |
| structural_family_only | 20 | 0.5631 | 30 | 8 | 3 |
