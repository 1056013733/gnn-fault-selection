from __future__ import annotations

import csv
import json
from pathlib import Path


def test_epfl20_current_main_selector_meets_acceptance_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (
            root
            / "analysis"
            / "v38_no_hand_parameters_epfl20_20260526_01"
            / "acceptance_summary.json"
        ).read_text(encoding="utf-8")
    )
    main = summary["epfl20_random_vector_main"]

    assert main["loss_rows"] == 0
    assert main["random_gate_fail"] == 0
    assert main["macro_ideal_ratio_raw"] > main["static_baseline_macro_ideal_ratio_raw"]


def test_iscas85_cross_benchmark_selector_meets_acceptance_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads(
        (root / "analysis" / "v38_no_hand_parameters_iscas85_20260526_01" / "summary.json").read_text(encoding="utf-8")
    )

    assert summary["combined_loss_rows"] == 0
    assert summary["combined_random_gate_fail"] == 0
    assert summary["combined_macro_ideal_ratio_raw"] >= 0.89


def test_iscas85_key_family_choices_are_stable() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = {
        row["circuit"]: row
        for row in csv.DictReader(
            (root / "analysis" / "v38_no_hand_parameters_iscas85_20260526_01" / "circuits.csv").open()
        )
    }

    assert rows["c6288"]["chosen_family"] == "mass_balance"
    for circuit in ["c432", "c1908", "c2670", "c3540", "c5315", "c7552"]:
        assert rows[circuit]["chosen_family"] == "dist_avg_inv"
    for circuit in ["c499", "c880", "c1355"]:
        assert rows[circuit]["chosen_family"] == "pdom_dist"
