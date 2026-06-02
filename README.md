# SEGR Random-Vector Reproduction Package

This package follows the same top-level format as the earlier random-vector
release package, but uses the cleaned Structure-Enhanced Graph Ranking (SEGR)
selector names in code-facing entry points. Generated result directories retain
their internal run identifiers so the experiment provenance remains auditable.

It keeps the formal reproduction materials:

- EPFL20 random-vector result summaries and raw injection inputs.
- ISCAS85 direct generalization check.
- Current selector/source code.
- Feature caches, `*_fi.v` files, and full-injection JSON files needed for
  offline reproduction.
- Verification tests and analysis tables.

## Main EPFL20 Random-Vector Results

These are the main EPFL20 results for the paper-style random-vector protocol.
They are evaluated from the current `node_debug.csv` `global_rank` /
`chosen_family` outputs, not bridged from an older release directory.

1. `analysis/v38_single_seed_main_7089_20260526_01/`
   - EPFL20 selected seed `7089`, 128 vectors per circuit.
   - Macro ideal ratio: `0.8143664407170059`.
   - Loss rows: `0`.
   - Random gate failures: `0`.
   - Static baseline macro ideal ratio: `0.6678451169158429`.

The independent 5-seed stability archive is kept under
`analysis/segr_epfl20_5seed_128vectors_20260526_01/`.

## SEGR Selector Evidence

1. `analysis/v38_no_hand_parameters_epfl20_20260526_01/`
   - SEGR selector acceptance and current EPFL20 selected-seed check.
   - `acceptance_summary.json`: records no-hand-parameter selector constraints.
   - The controlled/FI regression value is not the main random-vector EPFL20
     result; it is retained only as an auxiliary diagnostic.

2. `analysis/v38_no_hand_parameters_iscas85_20260526_01/`
   - ISCAS85 direct generalization check using the same unified selector.
   - Combined macro ideal ratio: `0.8909971101470905`.
   - Combined macro closure: `0.4300365930501083`.
   - Loss rows: `0`.
   - Random gate failures: `0`.

## Method/Protocol Notes

- One global GNN ranking is fixed per circuit; budgets 5%, 10%, and 20% are
  prefix slices of that ranking.
- The runtime selector does not read FI, oracle, random-vector results,
  `name_len`, gate type, signal direction, RTL tokens, AST/DFG, or semantic
  features.
- SEGR removes the remaining hand-written selector parameters from the cache
  structural signal, fusion, frontier, and arbitration path.
- The cache structural signal is computed from circuit-local structural rank
  geometry instead of fixed per-feature coefficients.
- GNN/structure weights are adapted from circuit-local rank geometry.
- Frontier/core/tail statistics are derived from runtime-visible non-semantic
  structural feature ranks through structural effective dimension.
- Raw FI/oracle data are included only for offline evaluation and metric
  reconstruction.

## Included Code

- `standalone/global_rank.py`: SEGR selector.
- `standalone/gnn_rank.py`: GNN ranking generation.
- `standalone/run_epfl20.py`: EPFL20 controlled/FI style runner.
- `scripts/run_epfl20_vector_stability.py`: EPFL20 random-vector stability
  evaluation.
- `scripts/merge_epfl_vector_count_shards.py`: shard validation/merge helper.
- `scripts/materialize_reviewer_experiments.py`: selected-seed reviewer table
  materialization for main, supervised LOCO, component, FuSa, and architecture
  rows.
- `scripts/build_rank_diagnostics.py`: rank-space Taylor diagnostics and
  budgeted-gain distribution data for Fig. 4/5.
- `scripts/epfl_random_vector_helpers.py`: shared EPFL random-vector evaluation
  helpers.
- `scripts/run_iscas85_89_main.py`: external benchmark runner used for the
  ISCAS85 check.

## Included Data

- `data/Supplementary_Experiments/feature_cache/*.pkl`: EPFL20 feature caches.
- `data/Supplementary_Experiments/full_injection_results_verilator/<circuit>/full_injection_results.json`:
  raw EPFL full-injection JSON.
- `data/Supplementary_Experiments/full_injection_results_verilator/<circuit>/*_fi.v`:
  gate-level random-vector simulation netlists.
- `data/external_benchmarks/iscas85/`: ISCAS85 benchmark inputs for the external
  generalization check.

## Verification

From the package root:

```powershell
python -m pytest tests -q
python -m py_compile standalone\global_rank.py standalone\run_epfl20.py scripts\run_epfl20_vector_stability.py scripts\run_iscas85_89_main.py scripts\materialize_reviewer_experiments.py scripts\build_rank_diagnostics.py scripts\epfl_random_vector_helpers.py
```

Expected test result:

```text
21 passed
```
