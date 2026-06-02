# Missing Comparison Experiments

These rows supplement reviewer-requested comparison baselines without changing the SEGR ranking path.
All methods produce one fixed target-circuit order before FI counts, RV-Oracle counts, random-vector outcomes, held-out labels, or evaluation metrics are opened.

The SCOAP/testability and observability rows are explicitly reported as runtime-visible structural proxies, not as independent commercial-tool SCOAP measurements.

## Method Summary

| Method | Circuits | Macro ideal | Loss rows | Random gate fails | >=0.75 no-loss circuits |
| --- | ---: | ---: | ---: | ---: | ---: |
| centrality_only | 20 | 0.4300 | 38 | 15 | 3 |
| cone_centrality_proxy | 20 | 0.5647 | 35 | 8 | 4 |
| equal_rank_fusion_static_cache_gnn | 20 | 0.6103 | 32 | 6 | 3 |
| max_visible_signal | 20 | 0.5206 | 34 | 10 | 3 |
| observability_cone_proxy | 20 | 0.7183 | 17 | 5 | 6 |
| rrf_static_cache_gnn | 20 | 0.6107 | 33 | 6 | 3 |
| scoap_proxy | 20 | 0.7388 | 7 | 4 | 11 |
| structural_borda_no_gnn | 20 | 0.6173 | 31 | 8 | 3 |

## Paired SEGR Advantage

| Method | Rows | Mean delta | 95% bootstrap CI | Positive | Negative | Zero |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| centrality_only | 60 | 0.3843 | [0.3158, 0.4540] | 55 | 0 | 5 |
| cone_centrality_proxy | 60 | 0.2497 | [0.1851, 0.3152] | 47 | 7 | 6 |
| equal_rank_fusion_static_cache_gnn | 60 | 0.2041 | [0.1549, 0.2566] | 52 | 5 | 3 |
| max_visible_signal | 60 | 0.2937 | [0.2361, 0.3537] | 52 | 3 | 5 |
| observability_cone_proxy | 60 | 0.0961 | [0.0617, 0.1330] | 40 | 13 | 7 |
| rrf_static_cache_gnn | 60 | 0.2036 | [0.1696, 0.2391] | 57 | 0 | 3 |
| scoap_proxy | 60 | 0.0756 | [0.0470, 0.1098] | 38 | 2 | 20 |
| structural_borda_no_gnn | 60 | 0.1970 | [0.1497, 0.2467] | 51 | 6 | 3 |
