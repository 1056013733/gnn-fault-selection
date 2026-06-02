from __future__ import annotations

import ast
import inspect

from standalone import global_rank


SELECTOR_DECISION_FUNCTIONS = [
    global_rank.auto_distance_guard_ready,
    global_rank.choose_family,
]


def _numeric_threshold_compares(fn) -> list[tuple[int, str]]:
    tree = ast.parse(inspect.getsource(fn))
    offenders: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, (int, float)):
                if float(comparator.value) not in {0.0, 1.0}:
                    offenders.append((node.lineno, ast.unparse(node)))
    return offenders


def test_selector_family_arbitration_has_no_handwritten_numeric_thresholds() -> None:
    offenders: list[tuple[str, int, str]] = []
    for fn in SELECTOR_DECISION_FUNCTIONS:
        offenders.extend((fn.__name__, line, expr) for line, expr in _numeric_threshold_compares(fn))

    assert offenders == []


def test_build_global_rank_does_not_use_external_threshold_parameters() -> None:
    source = inspect.getsource(global_rank.build_global_rank)
    body = source[source.index(") -> RankResult:") :]

    for forbidden in [
        "gnn_nonrandom_min",
        "gnn_struct_agree_min",
        "static_protection_start",
        "static_protection_power",
    ]:
        assert forbidden not in body


def test_selector_is_gnn_required_without_external_inputs() -> None:
    source = inspect.getsource(global_rank)

    for required in ["gnn_rank", "gnn_frontier_overlap", "final_rank"]:
        assert required in source

    for forbidden in [
        "selector_prototypes",
        "SELECTOR_PROTOTYPE_PATH",
        "load_selector_prototypes",
        "read_text",
        "json.loads",
        "oracle",
        "random_vector",
        "vector_summary",
        "name_len",
    ]:
        assert forbidden not in source

