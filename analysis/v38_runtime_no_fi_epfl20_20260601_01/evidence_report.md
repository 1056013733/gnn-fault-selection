# v38 Runtime-No-FI EPFL20 Evidence

## Checks

- runtime_no_fi in debug/run_config.json: True.
- Results rows: 120; perf rows: 20.
- All result rows mark runtime-no-FI: True.
- All perf rows mark runtime-no-FI: True.
- Controlled FI-JSON macro fault_instance_nsde: 0.652801777968.
- Controlled loss rows versus Static-Prox: 0.
- Rank-hash equality versus current v38 debug rankings: 20/20.
- Family equality versus current v38 debug rankings: 20/20.
- Family-reason equality versus current v38 debug rankings: 20/20.

## Interpretation

The runtime-no-FI execution reconstructs EPFL20 rankings with load_fi=False in the ranking path and is compared directly against the current v38 debug outputs. No older rank-hash bridge is used.

The controlled FI-JSON rows are a regression/evidence table, not the selected-seed random-vector main table. The paper-facing selected-seed result is generated directly in analysis/v38_single_seed_main_7089_20260526_01/.

## Files

- outputs_runs/v38_runtime_no_fi_epfl20_20260601_01/results.csv
- outputs_runs/v38_runtime_no_fi_epfl20_20260601_01/perf.csv
- outputs_runs/v38_runtime_no_fi_epfl20_20260601_01/debug/*.csv
- analysis/v38_runtime_no_fi_epfl20_20260601_01/rank_hash_compare_vs_current_v38.csv
- analysis/v38_runtime_no_fi_epfl20_20260601_01/file_manifest.csv
- analysis/v38_runtime_no_fi_epfl20_20260601_01/evidence_summary.json
