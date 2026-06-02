# SEGR Unified Reviewer Experiment Index

Selected main seed: `7089`; vectors: `128`.

FI-JSON reconstruction is not used for the paper-facing SEGR main result.
The SEGR unsupervised selector runtime does not read FI/oracle/random-vector counts; those counts are used only for offline evaluation and supervised training labels.

| Experiment | Output | Exists |
| --- | --- | ---: |
| main | `analysis\v38_single_seed_main_7089_20260526_01\main_seed_summary.csv` | True |
| same_feature | `analysis\v38_same_feature_baselines_seed7089_20260526_01\baseline_method_summary.csv` | True |
| architecture | `analysis\v38_architecture_ablation_seed7089_20260526_01\architecture_ablation_summary.csv` | True |
| component | `analysis\v38_component_ablation_seed7089_20260526_01\component_ablation_summary.csv` | True |
| fusa | `analysis\v38_fusa_supervised_fair_seed7089_20260526_01\fusa_method_summary.csv` | True |
