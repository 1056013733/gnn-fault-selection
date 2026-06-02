from __future__ import annotations

import inspect
from pathlib import Path

from standalone import run_epfl20
from standalone.global_rank import (
    CANDIDATE_FAMILIES,
    auto_distance_guard_ready,
    build_global_rank,
    choose_family,
)


def test_selector_is_gnn_aware_and_runtime_visible() -> None:
    source = inspect.getsource(choose_family)

    assert "gnn_frontier_overlap" in source
    assert "final_rank" in source
    assert "gnn_rank" not in source
    for forbidden in ["oracle", "load_fi", "random_vector", "vector_summary", "name_len"]:
        assert forbidden not in source


def test_ranker_removes_handwritten_selector_parameters() -> None:
    signature = inspect.signature(build_global_rank)
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


def test_runner_removes_handwritten_selector_flags() -> None:
    source = inspect.getsource(run_epfl20)
    for flag in [
        "--residual-scale",
        "--gnn-weight",
        "--struct-weight",
        "--gnn-nonrandom-min",
        "--gnn-struct-agree-min",
        "--static-protection-start",
        "--static-protection-power",
    ]:
        assert flag not in source


def test_selector_has_no_external_prototype_or_file_input() -> None:
    source = inspect.getsource(__import__("standalone.global_rank", fromlist=["*"]))

    for forbidden in [
        "SELECTOR_PROTOTYPE_PATH",
        "selector_prototypes",
        "load_selector_prototypes",
        "choose_family_calibrated",
        "read_text",
        "json.loads",
    ]:
        assert forbidden not in source

    standalone_dir = Path(__file__).resolve().parents[1] / "standalone"
    assert not list(standalone_dir.glob("selector_prototypes*.json"))
    assert not list(standalone_dir.glob("*.bak*"))

