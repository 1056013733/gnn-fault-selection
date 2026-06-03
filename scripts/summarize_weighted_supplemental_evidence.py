#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fnum(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def circuit_weights(main_circuits: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in read_csv(main_circuits):
        out[row["circuit"]] = fnum(row.get("debug_rows"))
    return out


def summarize_circuit_methods(circuit_rows: list[dict[str, str]], weights: dict[str, float]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in circuit_rows:
        by_method[row["method"]].append(row)

    out: list[dict[str, Any]] = []
    for method, rows in sorted(by_method.items()):
        vals = [fnum(row["method_ratio"]) for row in rows]
        wsum = sum(weights.get(row["circuit"], 0.0) for row in rows)
        weighted = (
            sum(weights.get(row["circuit"], 0.0) * fnum(row["method_ratio"]) for row in rows) / wsum
            if wsum
            else 0.0
        )
        out.append(
            {
                "method": method,
                "circuits": len(rows),
                "macro_ideal_ratio": mean(vals) if vals else 0.0,
                "node_weighted_ideal_ratio": weighted,
                "loss_rows": sum(int(fnum(row.get("loss_rows"))) for row in rows),
                "random_failures": sum(1 for row in rows if not truthy(row.get("random_passed"))),
                "ge_075_circuits": sum(1 for row in rows if fnum(row["method_ratio"]) >= 0.75),
                "eligible_nodes": int(wsum),
            }
        )
    return out


def summarize_absolute_counts(row_paths: list[Path], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, str]] = []
    for path in row_paths:
        if path.exists():
            rows.extend(read_csv(path))
    by_method_budget: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_method_budget[(row.get("method", label), row["budget"])].append(row)

    out: list[dict[str, Any]] = []
    for (method, budget), items in sorted(by_method_budget.items()):
        h_method = sum(fnum(row.get("method_value")) for row in items)
        h_static = sum(fnum(row.get("static_value")) for row in items)
        oracle = sum(fnum(row.get("oracle_value")) for row in items)
        out.append(
            {
                "source": label,
                "method": method,
                "budget": budget,
                "H_method": h_method,
                "H_static": h_static,
                "O": oracle,
                "oracle_count_weighted_ideal_ratio": h_method / oracle if oracle else 0.0,
                "static_oracle_count_weighted_ratio": h_static / oracle if oracle else 0.0,
                "rows": len(items),
            }
        )
    return out


def summarize_vector_dir(root: Path, dirname: str, vectors: int, weights: dict[str, float]) -> dict[str, Any]:
    path = root / "analysis" / dirname / "vector_circuits.csv"
    if not path.exists():
        return {
            "vectors": vectors,
            "status": "pending_missing_vector_circuits",
            "circuits": 0,
            "macro_ideal_ratio": "",
            "node_weighted_ideal_ratio": "",
            "loss_rows": "",
            "random_failures": "",
            "eligible_nodes": "",
            "source_dir": dirname,
        }
    rows = read_csv(path)
    vals = [fnum(row["seed_ratio_mean"]) for row in rows]
    wsum = sum(weights.get(row["circuit"], fnum(row.get("candidate_nodes"))) for row in rows)
    weighted = sum(
        weights.get(row["circuit"], fnum(row.get("candidate_nodes"))) * fnum(row["seed_ratio_mean"])
        for row in rows
    ) / wsum
    return {
        "vectors": vectors,
        "status": "complete",
        "circuits": len(rows),
        "macro_ideal_ratio": mean(vals) if vals else 0.0,
        "node_weighted_ideal_ratio": weighted,
        "loss_rows": sum(int(fnum(row.get("loss_rows_total"))) for row in rows),
        "random_failures": sum(int(fnum(row.get("random_gate_fail_total"))) for row in rows),
        "eligible_nodes": int(wsum),
        "source_dir": dirname,
    }


def summarize_runtime(perf_path: Path, weights: dict[str, float]) -> list[dict[str, Any]]:
    rows = read_csv(perf_path)
    cols = [
        ("feature_seconds", "load_seconds"),
        ("gnn_seconds", "gnn_seconds"),
        ("selector_seconds", "rank_seconds"),
        ("total_seconds", "total_seconds"),
    ]
    wsum = sum(weights.get(row["circuit"], 0.0) for row in rows)
    out: list[dict[str, Any]] = []
    for out_name, col in cols:
        vals = [fnum(row[col]) for row in rows]
        out.append(
            {
                "component": out_name,
                "sum_seconds": sum(vals),
                "mean_seconds_per_circuit": mean(vals) if vals else 0.0,
                "node_weighted_mean_seconds": (
                    sum(weights.get(row["circuit"], 0.0) * fnum(row[col]) for row in rows) / wsum if wsum else 0.0
                ),
                "max_seconds": max(vals) if vals else 0.0,
                "circuits": len(rows),
            }
        )
    return out


def markdown_table(rows: list[dict[str, Any]], keys: list[str]) -> list[str]:
    lines = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for row in rows:
        cells: list[str] = []
        for key in keys:
            val = row.get(key, "")
            if isinstance(val, float):
                cells.append(f"{val:.4f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/v38_weighted_supplemental_tables_20260603_01"))
    args = parser.parse_args()
    root = args.root.resolve()
    out_dir = root / args.out_dir

    weights = circuit_weights(root / "analysis/v38_single_seed_main_7089_20260526_01/main_seed_circuits.csv")

    vector_rows = [
        summarize_vector_dir(root, "v38_vector_count_sensitivity_32_seed7089_20260603_01", 32, weights),
        summarize_vector_dir(root, "v38_vector_count_sensitivity_64_seed7089_20260603_01", 64, weights),
        {
            "vectors": 128,
            "status": "complete",
            "circuits": 20,
            "macro_ideal_ratio": mean(
                fnum(row["method_ratio"])
                for row in read_csv(root / "analysis/v38_single_seed_main_7089_20260526_01/main_seed_circuits.csv")
            ),
            "node_weighted_ideal_ratio": sum(
                weights[row["circuit"]] * fnum(row["method_ratio"])
                for row in read_csv(root / "analysis/v38_single_seed_main_7089_20260526_01/main_seed_circuits.csv")
            )
            / sum(weights.values()),
            "loss_rows": sum(
                int(fnum(row["loss_rows"]))
                for row in read_csv(root / "analysis/v38_single_seed_main_7089_20260526_01/main_seed_circuits.csv")
            ),
            "random_failures": sum(
                1
                for row in read_csv(root / "analysis/v38_single_seed_main_7089_20260526_01/main_seed_circuits.csv")
                if not truthy(row["random_passed"])
            ),
            "eligible_nodes": int(sum(weights.values())),
            "source_dir": "v38_single_seed_main_7089_20260526_01",
        },
        summarize_vector_dir(root, "v38_vector_count_sensitivity_256_seed7089_20260603_02", 256, weights),
    ]

    scoap_rows = summarize_circuit_methods(
        read_csv(root / "analysis/v38_supplemental_selector_scoap_seed7089_20260602_01/supplemental_circuits.csv"),
        weights,
    )
    comparison_rows = summarize_circuit_methods(
        read_csv(root / "analysis/v38_missing_comparison_baselines_seed7089_20260602_01/comparison_circuits.csv"),
        weights,
    )
    runtime_rows = summarize_runtime(root / "outputs_runs/v38_runtime_no_fi_epfl20_20260601_01/perf.csv", weights)
    absolute_rows = summarize_absolute_counts(
        [root / "analysis/v38_supplemental_selector_scoap_seed7089_20260602_01/supplemental_rows.csv"],
        "epfl20_seed7089_128rv_scoap_supplemental",
    )

    write_csv(out_dir / "rv_count_sensitivity_weighted.csv", vector_rows)
    write_csv(out_dir / "scoap_co_weighted_summary.csv", scoap_rows)
    write_csv(out_dir / "structural_proxy_weighted_summary.csv", comparison_rows)
    write_csv(out_dir / "runtime_scaling_cost.csv", runtime_rows)
    write_csv(out_dir / "absolute_count_oracle_weighted.csv", absolute_rows)

    report: list[str] = [
        "# Weighted Supplemental Evidence Tables",
        "",
        "All weighted ideal ratios use EPFL20 eligible-node counts from the materialized seed-7089 main run.",
        "RV-count sensitivity reports complete EPFL20 rows for 32, 64, 128, and 256 vectors.",
        "",
        "## RV-count Sensitivity",
        "",
        *markdown_table(
            vector_rows,
            [
                "vectors",
                "status",
                "circuits",
                "macro_ideal_ratio",
                "node_weighted_ideal_ratio",
                "loss_rows",
                "random_failures",
            ],
        ),
        "",
        "## Standard SCOAP / CO Baseline",
        "",
        *markdown_table(
            [
                row
                for row in scoap_rows
                if row["method"]
                in {
                    "segr_structure_derived_selector",
                    "standard_scoap_testability",
                    "scoap_co_only",
                    "scoap_min_fault_cost",
                    "scoap_avg_fault_cost",
                    "scoap_worst_fault_cost",
                }
            ],
            ["method", "macro_ideal_ratio", "node_weighted_ideal_ratio", "loss_rows", "random_failures"],
        ),
        "",
        "## Runtime / Scaling Cost",
        "",
        *markdown_table(runtime_rows, ["component", "sum_seconds", "mean_seconds_per_circuit", "node_weighted_mean_seconds", "max_seconds"]),
        "",
        "## Absolute-count / Oracle-count Weighted",
        "",
        *markdown_table(
            [
                row
                for row in absolute_rows
                if row["method"] in {"segr_structure_derived_selector", "standard_scoap_testability", "scoap_co_only"}
            ],
            ["method", "budget", "H_method", "H_static", "O", "oracle_count_weighted_ideal_ratio"],
        ),
    ]
    (out_dir / "weighted_supplemental_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    manifest = {
        "out_dir": str(args.out_dir),
        "eligible_node_total": int(sum(weights.values())),
        "tables": [
            "rv_count_sensitivity_weighted.csv",
            "scoap_co_weighted_summary.csv",
            "structural_proxy_weighted_summary.csv",
            "runtime_scaling_cost.csv",
            "absolute_count_oracle_weighted.csv",
            "weighted_supplemental_report.md",
        ],
    }
    (out_dir / "summary_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
