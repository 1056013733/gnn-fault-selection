from __future__ import annotations

import inspect
import json
from pathlib import Path

from scripts import materialize_reviewer_experiments
from standalone import data_io, global_rank
from scripts import run_iscas85_89_main


def test_segr_selector_has_no_public_handwritten_selector_parameters() -> None:
    signature = inspect.signature(global_rank.build_global_rank)
    for name in [
        "residual_scale",
        "gnn_weight",
        "struct_weight",
        "gnn_nonrandom_min",
        "gnn_struct_agree_min",
        "static_protection_start",
        "static_protection_power",
    ]:
        assert name not in signature.parameters


def test_segr_frontier_is_structure_derived_not_fixed_percentile() -> None:
    source = inspect.getsource(global_rank.family_metrics)

    for forbidden in ["0.20", "0.10", "0.95", "candidate_ensemble"]:
        assert forbidden not in source

    assert "structural_frontier_profile" in source
    assert "frontier_fraction" in source
    assert "tail_quantile" in source


def test_selector_metric_names_do_not_imply_fixed_percentages() -> None:
    source = inspect.getsource(global_rank)

    for forbidden in [
        "peer20",
        "gnn20",
        "cache20",
        "static10",
        "tail95",
        "top20_mean",
        "family_peer20",
        "family_gnn20",
        "family_cache20",
        "family_static10",
        "family_tail95",
        "family_top20_mean",
    ]:
        assert forbidden not in source

    for required in [
        "peer_frontier_overlap",
        "gnn_frontier_overlap",
        "cache_frontier_overlap",
        "static_core_overlap",
        "tail_separation",
        "frontier_mean",
    ]:
        assert required in source


def test_cache_structural_score_has_no_fixed_feature_weights() -> None:
    source = inspect.getsource(data_io.load_circuit)
    source += inspect.getsource(run_iscas85_89_main.load_bench_circuit)

    for forbidden in ["0.16 *", "0.14 *", "0.10 *", "0.08 *", "0.55"]:
        assert forbidden not in source

    assert "adaptive_cache_struct_score" in source


def test_adaptive_cache_structural_score_ignores_name_length_artifact() -> None:
    names = ["n0", "n1", "n2", "n3"]
    feature_by_name = {
        "n0": {
            "pagerank": 0.0,
            "betweenness": 0.0,
            "dist_avg_inv": 0.0,
            "out_deg": 0.0,
            "name_len": 1000.0,
        },
        "n1": {
            "pagerank": 0.25,
            "betweenness": 0.25,
            "dist_avg_inv": 0.25,
            "out_deg": 0.25,
            "name_len": 0.0,
        },
        "n2": {
            "pagerank": 0.5,
            "betweenness": 0.5,
            "dist_avg_inv": 0.5,
            "out_deg": 0.5,
            "name_len": 0.0,
        },
        "n3": {
            "pagerank": 1.0,
            "betweenness": 1.0,
            "dist_avg_inv": 1.0,
            "out_deg": 1.0,
            "name_len": 0.0,
        },
    }

    scores = data_io.adaptive_cache_struct_score(names, feature_by_name)

    assert set(scores) == set(names)
    assert scores["n3"] > scores["n2"] > scores["n1"] > scores["n0"]


def test_wl_common_repair_requires_circuit_resolvable_scores() -> None:
    metrics = {
        "wl3_common_dist": {
            "peer_frontier_overlap": 0.3,
            "gnn_frontier_overlap": 0.1,
            "cache_frontier_overlap": 0.9,
            "static_core_overlap": 0.2,
            "tail_separation": 0.1,
            "frontier_mean": 0.1,
            "score_resolution": 0.005,
        },
        "pdom_dist": {
            "peer_frontier_overlap": 0.4,
            "gnn_frontier_overlap": 0.2,
            "cache_frontier_overlap": 0.1,
            "static_core_overlap": 0.8,
            "tail_separation": 0.0,
            "frontier_mean": 0.1,
            "score_resolution": 0.1,
        },
    }

    family, reason = global_rank.choose_family(metrics, n_nodes=10000)

    assert (family, reason) == ("pdom_dist", "relative-pdom-static-aligned")


def test_main_seed_materialization_uses_current_debug_context_not_five_seed_bridge() -> None:
    signature = inspect.signature(materialize_reviewer_experiments.materialize_main_seed)
    assert list(signature.parameters) == ["args", "context"]

    source = inspect.getsource(materialize_reviewer_experiments.materialize_main_seed)

    assert "five_seed_dir" not in source
    assert "read_debug_rows" in source
    assert "debug_rank" in source
    assert "eval_rank" in source


def test_segr_acceptance_summary_passes_epfl_and_iscas_gates() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (
            root
            / "analysis"
            / "v38_no_hand_parameters_epfl20_20260526_01"
            / "acceptance_summary.json"
        ).read_text(encoding="utf-8")
    )

    assert summary["removed_public_handwritten_selector_parameters"] is True
    assert summary["no_fi_or_oracle_or_random_vector_in_selector"] is True
    assert summary["structure_derived_frontier"] is True

    epfl = summary["epfl20_random_vector_main"]
    assert epfl["evidence_source"] == "current_debug_global_rank"
    assert epfl["loss_rows"] == 0
    assert epfl["random_gate_fail"] == 0
    assert epfl["macro_ideal_ratio_raw"] > epfl["static_baseline_macro_ideal_ratio_raw"]

    assert summary["iscas85_v38"]["combined_loss_rows"] == 0
    assert summary["iscas85_v38"]["combined_random_gate_fail"] == 0
    assert summary["iscas85_v38"]["combined_macro_ideal_ratio_raw"] >= 0.8909971101470905
