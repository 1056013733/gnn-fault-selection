from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_supplemental_experiments.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("materialize_supplemental_experiments", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scoap_costs_rank_observable_output_before_internal_node() -> None:
    mod = load_script_module()
    gates = [
        mod.Gate("AND2", "n1", ("a", "b")),
        mod.Gate("OR2", "y", ("n1", "c")),
    ]

    scores = mod.scoap_testability_scores(["n1", "y"], ["a", "b", "c"], ["y"], gates)

    assert scores["y"] > scores["n1"]


def test_scoap_variants_separate_optimistic_average_and_worst_costs() -> None:
    mod = load_script_module()
    gates = [
        mod.Gate("AND2", "n1", ("a", "b")),
        mod.Gate("OR2", "y", ("n1", "c")),
    ]

    variants = mod.scoap_testability_score_variants(["n1", "y"], ["a", "b", "c"], ["y"], gates)

    assert {
        "scoap_co_only",
        "scoap_min_fault_cost",
        "scoap_avg_fault_cost",
        "scoap_worst_fault_cost",
    }.issubset(variants)
    assert variants["scoap_min_fault_cost"]["n1"] >= variants["scoap_avg_fault_cost"]["n1"]
    assert variants["scoap_avg_fault_cost"]["n1"] >= variants["scoap_worst_fault_cost"]["n1"]


def test_script_declares_supplemental_boundaries() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "standard_scoap_testability" in source
    assert "scoap_avg_fault_cost" in source
    assert "selector_family_diagnostics" in source
    assert "target FI labels before ranking" in source
    assert "diagnostic only" in source


def test_eval_rank_accepts_cached_random_gate_stats(monkeypatch) -> None:
    mod = load_script_module()

    def fail_random_ratios(*_args, **_kwargs):
        raise AssertionError("random ratios should be cached per circuit")

    monkeypatch.setattr(mod, "random_ratios", fail_random_ratios)
    args = type("Args", (), {"vector_seed": 7089, "vectors": 128, "random_samples": 1000, "random_seed": 95289})()

    _rows, summary = mod.eval_rank(
        "toy",
        "toy_method",
        ["a", "b"],
        ["a", "b"],
        {"a": 2, "b": 1},
        ["a", "b"],
        args,
        random_gate_stats={"random_mean": 0.5, "random_p05": 0.25, "random_p95": 0.75, "gate": 0.5},
    )

    assert summary["random_mean"] == 0.5
    assert summary["random_passed"] is True
