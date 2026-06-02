from __future__ import annotations

import csv
import json
from pathlib import Path

from standalone import run_epfl20


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = ROOT / "analysis" / "v38_runtime_no_fi_epfl20_20260601_01"
OUTPUT_DIR = ROOT / "outputs_runs" / "v38_runtime_no_fi_epfl20_20260601_01"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def test_runtime_no_fi_results_have_clean_schema() -> None:
    rows = read_rows(OUTPUT_DIR / "results.csv")
    assert rows

    forbidden = {
        "static_ratio",
        "static_ratio_mode",
        "boundary_gap",
        "static_ambiguity",
        "pool_mode",
        "gnn_training_skipped",
        "rank_selector_reason",
    }
    assert forbidden.isdisjoint(rows[0])
    assert all(row["runtime_no_fi"] == "True" for row in rows)
    assert all(row["rank_action"] for row in rows)


def test_runtime_no_fi_rank_hash_bridge_is_complete() -> None:
    summary = json.loads((EVIDENCE_DIR / "evidence_summary.json").read_text(encoding="utf-8"))
    assert summary["runtime_no_fi_config"] is True
    assert summary["all_result_rows_runtime_no_fi_true"] is True
    assert summary["all_perf_rows_runtime_no_fi_true"] is True
    assert summary["rank_hash_equal_vs_current_v38"] == summary["rank_hash_total"] == 20
    assert summary["family_equal_vs_current_v38"] == summary["family_total"] == 20
    assert summary["reason_equal_vs_current_v38"] == summary["reason_total"] == 20


def test_runner_source_does_not_emit_legacy_placeholder_columns() -> None:
    source = Path(run_epfl20.__file__).read_text(encoding="utf-8")
    for forbidden in [
        '"static_ratio"',
        '"boundary_gap"',
        '"static_ambiguity"',
        '"pool_mode"',
        '"gnn_training_skipped"',
        '"rank_selector_reason"',
    ]:
        assert forbidden not in source
