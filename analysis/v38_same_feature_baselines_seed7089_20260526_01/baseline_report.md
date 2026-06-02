# Same-Feature Supervised Baselines

All baselines use the same runtime-visible SEGR node features and selected-seed random-vector labels from training circuits only.

| Method | Circuits | Macro ideal | Loss rows | Random gate fails | >=0.75 no-loss circuits |
| --- | ---: | ---: | ---: | ---: | ---: |
| logistic_regression_same_feature_loco | 20 | 0.6962 | 17 | 20 | 3 |
| mlp_same_feature_loco | 20 | 0.6345 | 33 | 20 | 3 |
| random_forest_same_feature_loco | 20 | 0.7039 | 25 | 20 | 2 |
