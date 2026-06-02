#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.materialize_reviewer_experiments import (  # noqa: E402
    debug_rank,
    no_gnn_rank,
    rank_from_scores,
    read_csv,
    read_debug_rows,
    shuffled_gnn_rank,
    structural_family_heuristic_rank,
)
from scripts.epfl_random_vector_helpers import stable_static_order  # noqa: E402
from standalone.data_io import EPFL20, load_circuit  # noqa: E402
from standalone.evaluate import write_rows  # noqa: E402


SEGR_METHOD = "segr_structure_derived_selector"
LEGACY_SEGR_METHOD = "v38_no_hand_parameters_selector"

PUBLIC_NAME = {
    SEGR_METHOD: "SEGR",
    LEGACY_SEGR_METHOD: "SEGR",
    "pure_static_proximity": "Static-Prox",
    "no_gnn": "SEGR w/o GNN",
    "gnn_only": "GNN-only",
    "shuffled_gnn": "Shuffled-GNN",
    "cache_structural_only": "Cache-Struct",
    "structural_family_only": "Struct-Family",
}

FIG_METHODS = [
    SEGR_METHOD,
    "pure_static_proximity",
    "no_gnn",
    "gnn_only",
    "shuffled_gnn",
    "cache_structural_only",
    "structural_family_only",
]


def load_counts(path: Path) -> dict[str, int]:
    return {row["node"]: int(float(row.get("count", 0) or 0)) for row in read_csv(path)}


def oracle_rank(counts: dict[str, int], names: list[str]) -> list[str]:
    return sorted(names, key=lambda node: (int(counts.get(node, 0) or 0), node), reverse=True)


def rank_percentiles(ranked: list[str], names: list[str]) -> dict[str, float]:
    n = len(names)
    if n <= 1:
        return {name: 1.0 for name in names}
    position = {name: idx for idx, name in enumerate(ranked)}
    denom = float(n - 1)
    return {name: 1.0 - float(position[name]) / denom for name in names}


def centered_stats(method_vals: np.ndarray, oracle_vals: np.ndarray) -> dict[str, float]:
    method_centered = method_vals - float(np.mean(method_vals))
    oracle_centered = oracle_vals - float(np.mean(oracle_vals))
    method_std = float(np.sqrt(np.mean(method_centered ** 2)))
    oracle_std = float(np.sqrt(np.mean(oracle_centered ** 2)))
    if method_std <= 0.0 or oracle_std <= 0.0:
        corr = 0.0
    else:
        corr = float(np.mean(method_centered * oracle_centered) / (method_std * oracle_std))
    corr = max(-1.0, min(1.0, corr))
    crmse = float(np.sqrt(np.mean((method_centered - oracle_centered) ** 2)))
    return {
        "correlation": corr,
        "centered_rmse": crmse,
        "method_std": method_std,
        "oracle_std": oracle_std,
        "std_ratio": method_std / oracle_std if oracle_std else 0.0,
        "centered_rmse_norm": crmse / oracle_std if oracle_std else 0.0,
    }


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def build_rankings(args: argparse.Namespace, circuit: str, data: Any) -> dict[str, list[str]]:
    debug_rows = read_debug_rows(args.default_debug_dir, circuit)
    segr_rank, _chosen = debug_rank(args.default_debug_dir, circuit)
    return {
        SEGR_METHOD: segr_rank,
        LEGACY_SEGR_METHOD: segr_rank,
        "pure_static_proximity": stable_static_order(data),
        "no_gnn": no_gnn_rank(debug_rows),
        "gnn_only": rank_from_scores(debug_rows, "gnn_rank"),
        "shuffled_gnn": shuffled_gnn_rank(debug_rows, args.vector_seed, circuit),
        "cache_structural_only": rank_from_scores(debug_rows, "cache_struct_rank"),
        "structural_family_only": structural_family_heuristic_rank(data),
    }


def materialize_fig4(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_circuit: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for circuit in args.circuits:
        data = load_circuit(args.root, args.fi_root, circuit, load_fi=False)
        counts = load_counts(args.count_cache_dir / f"{circuit}_seed{args.vector_seed}_counts.csv")
        oracle_pct = rank_percentiles(oracle_rank(counts, data.node_names), data.node_names)
        oracle_arr = np.asarray([oracle_pct[name] for name in data.node_names], dtype=float)
        rankings = build_rankings(args, circuit, data)
        for method in FIG_METHODS:
            method_pct = rank_percentiles(rankings[method], data.node_names)
            method_arr = np.asarray([method_pct[name] for name in data.node_names], dtype=float)
            stats = centered_stats(method_arr, oracle_arr)
            row = {
                "figure": "Fig4",
                "circuit": circuit,
                "method": method,
                "public_name": PUBLIC_NAME[method],
                "n_nodes": len(data.node_names),
                **stats,
            }
            by_circuit.append(row)
            by_method[method].append(row)
            for name in data.node_names:
                long_rows.append(
                    {
                        "circuit": circuit,
                        "node": name,
                        "method": method,
                        "public_name": PUBLIC_NAME[method],
                        "oracle_percentile": oracle_pct[name],
                        "method_percentile": method_pct[name],
                        "detected_count": int(counts.get(name, 0) or 0),
                    }
                )
    summary: list[dict[str, Any]] = []
    for method in FIG_METHODS:
        rows = by_method[method]
        summary.append(
            {
                "figure": "Fig4",
                "method": method,
                "public_name": PUBLIC_NAME[method],
                "circuits": len(rows),
                "mean_correlation": mean(float(row["correlation"]) for row in rows),
                "mean_centered_rmse": mean(float(row["centered_rmse"]) for row in rows),
                "mean_centered_rmse_norm": mean(float(row["centered_rmse_norm"]) for row in rows),
                "mean_method_std": mean(float(row["method_std"]) for row in rows),
                "mean_oracle_std": mean(float(row["oracle_std"]) for row in rows),
                "mean_std_ratio": mean(float(row["std_ratio"]) for row in rows),
            }
        )
    return summary, by_circuit, long_rows


def materialize_fig5(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = read_csv(args.component_rows)
    main_method = SEGR_METHOD if any(row.get("method") == SEGR_METHOD for row in rows) else LEGACY_SEGR_METHOD
    by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        by_key[(row["circuit"], str(row["budget"]), row["method"])] = row
    detail: list[dict[str, Any]] = []
    variants = [method for method in FIG_METHODS if method != SEGR_METHOD]
    for circuit in args.circuits:
        budgets = sorted(
            {
                str(row["budget"])
                for row in rows
                if row.get("circuit") == circuit and row.get("method") == main_method
            },
            key=lambda x: float(x),
        )
        for budget in budgets:
            segr = by_key[(circuit, budget, main_method)]
            segr_value = float(segr["method_value"])
            oracle = float(segr["oracle_value"])
            for variant in variants:
                other = by_key[(circuit, budget, variant)]
                variant_value = float(other["method_value"])
                delta = (segr_value - variant_value) / oracle if oracle else 0.0
                if delta > args.eps:
                    sign = "positive"
                elif delta < -args.eps:
                    sign = "negative"
                else:
                    sign = "zero"
                detail.append(
                    {
                        "figure": "Fig5",
                        "circuit": circuit,
                        "budget": float(budget),
                        "variant": variant,
                        "public_name": PUBLIC_NAME[variant],
                        "segr_value": segr_value,
                        "variant_value": variant_value,
                        "oracle_value": oracle,
                        "delta": delta,
                        "segr_ratio": float(segr["row_ideal_ratio_raw"]),
                        "variant_ratio": float(other["row_ideal_ratio_raw"]),
                        "sign": sign,
                    }
                )
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detail:
        by_variant[row["variant"]].append(row)
    summary: list[dict[str, Any]] = []
    for variant in variants:
        items = by_variant[variant]
        values = [float(row["delta"]) for row in items]
        summary.append(
            {
                "figure": "Fig5",
                "variant": variant,
                "public_name": PUBLIC_NAME[variant],
                "samples": len(items),
                "positive": sum(1 for row in items if row["sign"] == "positive"),
                "zero": sum(1 for row in items if row["sign"] == "zero"),
                "negative": sum(1 for row in items if row["sign"] == "negative"),
                "mean_delta": mean(values) if values else 0.0,
                "median_delta": median(values) if values else 0.0,
                "q25_delta": quantile(values, 0.25),
                "q75_delta": quantile(values, 0.75),
                "min_delta": min(values) if values else 0.0,
                "max_delta": max(values) if values else 0.0,
            }
        )
    return summary, detail


def write_report(args: argparse.Namespace, fig4_summary: list[dict[str, Any]], fig5_summary: list[dict[str, Any]]) -> None:
    lines = [
        "# SEGR Fig. 4/5 Diagnostic Data",
        "",
        f"Suite: EPFL20; vector seed: `{args.vector_seed}`; vectors: `{args.vectors}`; budgets: 5%, 10%, and 20%.",
        "",
        "Fig. 4 is a rank-profile diagnostic against the random-vector oracle ranking. It is not the primary metric.",
        "Fig. 5 is a circuit-budget gain distribution derived from Table 4 component-ablation rows.",
        "",
        "Important: SEGR uses `global_rank` / `chosen_family` from `node_debug.csv`; `final_score` is an intermediate residual score and is not re-sorted here.",
        "",
        "## Fig. 4 Taylor Summary",
        "",
        "| Method | Corr. | Centered RMSE | Norm. CRMSE | Std. ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in fig4_summary:
        lines.append(
            f"| {row['public_name']} | {float(row['mean_correlation']):.4f} | "
            f"{float(row['mean_centered_rmse']):.4f} | {float(row['mean_centered_rmse_norm']):.4f} | "
            f"{float(row['mean_std_ratio']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Fig. 5 Gain Distribution Summary",
            "",
            "| Variant | Positive | Zero | Negative | Mean delta | Median delta | Q25 | Q75 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fig5_summary:
        lines.append(
            f"| {row['public_name']} | {row['positive']} | {row['zero']} | {row['negative']} | "
            f"{float(row['mean_delta']):.4f} | {float(row['median_delta']):.4f} | "
            f"{float(row['q25_delta']):.4f} | {float(row['q75_delta']):.4f} |"
        )
    (args.output_dir / "fig4_fig5_diagnostic_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(args: argparse.Namespace, fig4_summary: list[dict[str, Any]], fig5_detail: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        (args.output_dir / "plot_warning.txt").write_text(f"Plot generation skipped: {exc}\n", encoding="utf-8")
        return

    fig = plt.figure(figsize=(6.2, 4.8))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_rlim(0, 1.25)
    ax.set_title("Rank-space Taylor diagnostic", pad=18)
    ax.scatter([0.0], [1.0], marker="*", s=120, label="Oracle", color="black")
    for row in fig4_summary:
        corr = max(-1.0, min(1.0, float(row["mean_correlation"])))
        theta = math.acos(corr)
        radius = float(row["mean_std_ratio"])
        ax.scatter([theta], [radius], s=55, label=str(row["public_name"]))
    ax.legend(loc="upper right", bbox_to_anchor=(1.55, 1.10), fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig4_rank_space_taylor_diagnostic.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    variants = [method for method in FIG_METHODS if method != SEGR_METHOD]
    values_by_variant = [
        [float(row["delta"]) for row in fig5_detail if row["variant"] == variant]
        for variant in variants
    ]
    labels = [PUBLIC_NAME[variant] for variant in variants]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.boxplot(values_by_variant, tick_labels=labels, showfliers=False)
    rng = np.random.default_rng(12345)
    for idx, values in enumerate(values_by_variant, start=1):
        jitter = rng.normal(0.0, 0.045, size=len(values))
        ax.scatter(np.full(len(values), idx) + jitter, values, s=14, alpha=0.65)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_ylabel("Delta = (H_SEGR - H_variant) / O")
    ax.set_title("Budgeted gain distribution")
    ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "fig5_budgeted_gain_distribution.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--circuits", nargs="+", default=EPFL20)
    parser.add_argument("--vector-seed", type=int, default=7089)
    parser.add_argument("--vectors", type=int, default=128)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--default-debug-dir", type=Path, default=Path("outputs_runs/v38_no_hand_parameters_epfl20_20260526_01/debug"))
    parser.add_argument("--count-cache-dir", type=Path, default=Path("analysis/v38_single_seed_7089_count_cache_20260526_01"))
    parser.add_argument("--component-rows", type=Path, default=Path("analysis/v38_component_ablation_seed7089_20260526_01/component_ablation_rows.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/v38_fig4_fig5_diagnostics_seed7089_20260527_01"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig4_summary, fig4_by_circuit, fig4_long = materialize_fig4(args)
    fig5_summary, fig5_detail = materialize_fig5(args)

    write_rows(args.output_dir / "fig4_taylor_summary.csv", fig4_summary)
    write_rows(args.output_dir / "fig4_taylor_by_circuit.csv", fig4_by_circuit)
    write_rows(args.output_dir / "fig4_rank_percentiles_long.csv", fig4_long)
    write_rows(args.output_dir / "fig5_gain_distribution_summary.csv", fig5_summary)
    write_rows(args.output_dir / "fig5_gain_distribution_points.csv", fig5_detail)
    write_report(args, fig4_summary, fig5_summary)
    write_plots(args, fig4_summary, fig5_detail)
    print(f"Wrote Fig. 4/5 diagnostic data to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
