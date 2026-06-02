# SEGR Architecture Ablation

Only layers or hidden dimension is changed; all evaluation rows use the same selected random-vector oracle.

| Method | Circuits | Macro ideal | Loss rows | Random gate fails | >=0.75 no-loss circuits |
| --- | ---: | ---: | ---: | ---: | ---: |
| segr_h128 | 20 | 0.7888 | 1 | 2 | 12 |
| segr_h32 | 20 | 0.7561 | 4 | 2 | 11 |
| segr_l1 | 20 | 0.8063 | 0 | 1 | 13 |
| segr_l3 | 20 | 0.7658 | 3 | 2 | 11 |
| segr_structure_derived_selector | 20 | 0.8144 | 0 | 0 | 13 |
