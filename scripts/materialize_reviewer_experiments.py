#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_epfl20_vector_stability import BUDGETS, EPS, random_ratios, row_ratio
from scripts.epfl_random_vector_helpers import (
    compute_or_load_counts,
    debug_rank,
    eval_array_predictions,
    eval_rank,
    make_feature_item,
    positive_labels,
    read_csv,
    read_json,
    sample_indices,
    stable_static_order,
    summarize_circuit_seed,
    write_json,
)
from standalone.data_io import EPFL20, load_circuit
from standalone.evaluate import eval_selection, oracle_fault_instances, topk, write_rows


METHOD_DEFAULT = "segr_structure_derived_selector"
METHODS_ARCH = [
    METHOD_DEFAULT,
    "segr_l1",
    "segr_l3",
    "segr_h32",
    "segr_h128",
]
METHODS_BASELINE = [
    "logistic_regression_same_feature_loco",
    "random_forest_same_feature_loco",
    "mlp_same_feature_loco",
]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = row.get(key, default)
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_perf(perf_path: Path) -> dict[str, dict[str, float]]:
    if not perf_path.exists():
        return {}
    out: dict[str, dict[str, float]] = {}
    for row in read_csv(perf_path):
        circuit = str(row.get("circuit", ""))
        out[circuit] = {
            "load_seconds": read_float(row, "load_seconds"),
            "gnn_seconds": read_float(row, "gnn_seconds"),
            "rank_seconds": read_float(row, "rank_seconds"),
            "total_seconds": read_float(row, "total_seconds"),
        }
    return out


def make_runtime_row(
    experiment: str,
    suite: str,
    seed: int,
    circuit: str,
    method: str,
    selector: float = 0.0,
    train: float = 0.0,
    inference: float = 0.0,
    offline: float = 0.0,
    evaluation: float = 0.0,
    total: float = 0.0,
    note: str = "",
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "suite": suite,
        "seed": seed,
        "circuit": circuit,
        "method": method,
        "selector_runtime_seconds": selector,
        "train_runtime_seconds": train,
        "inference_runtime_seconds": inference,
        "offline_fi_oracle_runtime_seconds": offline,
        "evaluation_runtime_seconds": evaluation,
        "total_wall_seconds": total,
        "runtime_note": note,
    }


def group_by_method(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("method", ""))].append(row)
    return grouped


def write_method_report(path: Path, title: str, rows: list[dict[str, Any]], note: str) -> None:
    lines = [
        f"# {title}",
        "",
        note,
        "",
        "| Method | Circuits | Macro ideal | Loss rows | Random gate fails | >=0.75 no-loss circuits |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('method', '')} | {row.get('circuits', '')} | "
            f"{float(row.get('macro_ideal_ratio_raw', 0.0)):.4f} | "
            f"{row.get('loss_rows', '')} | {row.get('random_gate_fail', '')} | "
            f"{row.get('ideal_ratio_075_circuits', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def materialize_main_seed(args: argparse.Namespace, context: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = ensure_dir(args.analysis_root / f"v38_single_seed_main_{args.vector_seed}_20260526_01")
    rows: list[dict[str, Any]] = []
    circuits: list[dict[str, Any]] = []
    for circuit in args.circuits:
        data = context[circuit]["data"]
        counts = context[circuit]["counts"]
        static_order = context[circuit]["static_order"]
        debug_rows = read_debug_rows(args.default_debug_dir, circuit)
        full_rank, chosen_family = debug_rank(args.default_debug_dir, circuit)
        detail, summary = eval_rank(circuit, METHOD_DEFAULT, full_rank, static_order, counts, data.node_names, args)
        for row in detail:
            row["chosen_family"] = chosen_family
        summary["chosen_family"] = chosen_family
        summary["debug_rows"] = len(debug_rows)
        rows.extend(detail)
        circuits.append(summary)
    seed_rows = summarize_circuit_seed(rows, circuits)
    for row in seed_rows:
        row["vectors"] = int(args.vectors)
        row["selected_main_seed"] = True
    write_rows(out_dir / "main_seed_rows.csv", rows)
    write_rows(out_dir / "main_seed_circuits.csv", circuits)
    write_rows(out_dir / "main_seed_summary.csv", seed_rows)
    perf = read_perf(args.default_perf_csv)
    runtime_rows: list[dict[str, Any]] = []
    for circuit_row in circuits:
        circuit = str(circuit_row["circuit"])
        p = perf.get(circuit, {})
        count_meta = context[circuit].get("count_meta", {})
        runtime_rows.append(
            make_runtime_row(
                "single_seed_main",
                "epfl20",
                args.vector_seed,
                circuit,
                METHOD_DEFAULT,
                selector=p.get("total_seconds", 0.0),
                offline=read_float(count_meta, "seconds"),
                total=p.get("total_seconds", 0.0) + read_float(count_meta, "seconds"),
                note="selector runtime from perf.csv; offline runtime is selected-seed random-vector count/evaluation seconds from the current count cache.",
            )
        )
    write_rows(out_dir / "runtime_summary.csv", runtime_rows)
    report_lines = [
        "# SEGR Single-Seed Main Table",
        "",
        f"Selected main seed: `{args.vector_seed}`; vectors per circuit: `{args.vectors}`; suite: EPFL20.",
        "",
        "Rows are evaluated from the current debug ranking outputs for the selected seed.",
        "",
        "| Method | Circuits | Macro ideal | Loss rows | Random gate fails | >=0.75 no-loss circuits |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in seed_rows:
        report_lines.append(
            f"| {row['method']} | {row['circuits']} | {float(row['macro_ideal_ratio_raw']):.4f} | "
            f"{row['loss_rows']} | {row['random_gate_fail']} | {row['ideal_ratio_075_circuits']} |"
        )
    (out_dir / "main_seed_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return runtime_rows


def build_epfl_context(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    context: dict[str, Any] = {}
    count_manifest: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for circuit in args.circuits:
        data = load_circuit(args.root, args.fi_root, circuit, load_fi=False)
        counts, meta = compute_or_load_counts(args, circuit, data)
        static_order = stable_static_order(data)
        feature_item = make_feature_item(args, circuit, data, counts)
        context[circuit] = {
            "data": data,
            "counts": counts,
            "static_order": static_order,
            "feature_item": feature_item,
            "count_meta": meta,
        }
        count_manifest.append({"circuit": circuit, **meta})
    return context, count_manifest, time.perf_counter() - t0


def fit_predict_lr(x_train: np.ndarray, y_train_rank: np.ndarray, x_test: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=args.lr_max_iter,
            class_weight="balanced",
            random_state=args.seed,
            solver="lbfgs",
        ),
    )
    t0 = time.perf_counter()
    model.fit(x_train, positive_labels(y_train_rank, args.lr_positive_quantile))
    train_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = model.predict_proba(x_test)[:, 1].astype(np.float64, copy=False)
    infer_seconds = time.perf_counter() - t1
    return pred, train_seconds, infer_seconds


def fit_predict_rf(x_train: np.ndarray, y_train_rank: np.ndarray, x_test: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float, float]:
    model = RandomForestRegressor(
        n_estimators=args.rf_trees,
        max_depth=args.rf_max_depth if args.rf_max_depth > 0 else None,
        min_samples_leaf=args.rf_min_samples_leaf,
        n_jobs=args.rf_jobs,
        random_state=args.seed,
    )
    t0 = time.perf_counter()
    model.fit(x_train, y_train_rank)
    train_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = model.predict(x_test).astype(np.float64, copy=False)
    infer_seconds = time.perf_counter() - t1
    return pred, train_seconds, infer_seconds


def fit_predict_mlp(x_train: np.ndarray, y_train_rank: np.ndarray, x_test: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float, float]:
    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-4,
            max_iter=args.mlp_max_iter,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=args.seed,
        ),
    )
    t0 = time.perf_counter()
    model.fit(x_train, y_train_rank)
    train_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    pred = model.predict(x_test).astype(np.float64, copy=False)
    infer_seconds = time.perf_counter() - t1
    return pred, train_seconds, infer_seconds


def run_same_feature_loco(
    args: argparse.Namespace,
    context: dict[str, Any],
    *,
    methods: list[str],
    experiment_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for fold_idx, heldout in enumerate(args.circuits):
        x_parts = []
        y_parts = []
        sample_count = 0
        for circuit in args.circuits:
            if circuit == heldout:
                continue
            item = context[circuit]["feature_item"]
            idx = sample_indices(
                item["x"].shape[0],
                args.sample_per_circuit,
                args.seed + fold_idx * 1009 + sum(ord(ch) for ch in circuit),
            )
            x_parts.append(item["x"][idx])
            y_parts.append(item["y_rank"][idx])
            sample_count += int(idx.size)
        x_train = np.vstack(x_parts)
        y_train = np.concatenate(y_parts)
        x_test = context[heldout]["feature_item"]["x"]
        predictions: dict[str, tuple[np.ndarray, float, float]] = {}
        if "logistic_regression_same_feature_loco" in methods:
            predictions["logistic_regression_same_feature_loco"] = fit_predict_lr(x_train, y_train, x_test, args)
        if "random_forest_same_feature_loco" in methods:
            predictions["random_forest_same_feature_loco"] = fit_predict_rf(x_train, y_train, x_test, args)
        if "mlp_same_feature_loco" in methods:
            predictions["mlp_same_feature_loco"] = fit_predict_mlp(x_train, y_train, x_test, args)
        if "fusa_supervised_fair_mlp_loco" in methods:
            predictions["fusa_supervised_fair_mlp_loco"] = fit_predict_mlp(x_train, y_train, x_test, args)
        fold_common = {
            "fold": fold_idx,
            "heldout_circuit": heldout,
            "train_circuits": len(args.circuits) - 1,
            "train_samples": sample_count,
            "heldout_nodes": int(x_test.shape[0]),
        }
        for method, (pred, train_seconds, infer_seconds) in predictions.items():
            detail, circuit_summary = eval_array_predictions(
                heldout, method, pred, context[heldout]["feature_item"], args
            )
            rows.extend(detail)
            circuit_rows.append(circuit_summary)
            fold_rows.append(
                {
                    **fold_common,
                    "method": method,
                    "train_runtime_seconds": train_seconds,
                    "inference_runtime_seconds": infer_seconds,
                }
            )
            runtime_rows.append(
                make_runtime_row(
                    experiment_name,
                    "epfl20",
                    args.vector_seed,
                    heldout,
                    method,
                    train=train_seconds,
                    inference=infer_seconds,
                    evaluation=0.0,
                    total=train_seconds + infer_seconds,
                    note="LOCO supervised baseline; training labels come only from the other 19 circuits under the selected random-vector seed.",
                )
            )
        print(f"LOCO {fold_idx:02d} heldout={heldout} done", flush=True)
    return rows, circuit_rows, summarize_circuit_seed(rows, circuit_rows), fold_rows, runtime_rows


def materialize_baselines(args: argparse.Namespace, context: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = ensure_dir(args.analysis_root / f"v38_same_feature_baselines_seed{args.vector_seed}_20260526_01")
    t0 = time.perf_counter()
    rows, circuits, summary, folds, runtime_rows = run_same_feature_loco(
        args,
        context,
        methods=METHODS_BASELINE,
        experiment_name="same_feature_baselines",
    )
    wall = time.perf_counter() - t0
    for row in runtime_rows:
        row["total_wall_seconds"] = wall
    write_rows(out_dir / "baseline_rows.csv", rows)
    write_rows(out_dir / "baseline_circuits.csv", circuits)
    write_rows(out_dir / "baseline_method_summary.csv", summary)
    write_rows(out_dir / "baseline_folds.csv", folds)
    write_rows(out_dir / "runtime_summary.csv", runtime_rows)
    write_method_report(
        out_dir / "baseline_report.md",
        "Same-Feature Supervised Baselines",
        summary,
        "All baselines use the same runtime-visible SEGR node features and selected-seed random-vector labels from training circuits only.",
    )
    return runtime_rows


def materialize_architecture(args: argparse.Namespace, context: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = ensure_dir(args.analysis_root / f"v38_architecture_ablation_seed{args.vector_seed}_20260526_01")
    variants = [
        (METHOD_DEFAULT, args.default_debug_dir, args.default_perf_csv, 64, 2),
        ("segr_l1", args.variant_root / "layers1" / "debug", args.variant_root / "layers1" / "perf.csv", 64, 1),
        ("segr_l3", args.variant_root / "layers3" / "debug", args.variant_root / "layers3" / "perf.csv", 64, 3),
        ("segr_h32", args.variant_root / "hidden32" / "debug", args.variant_root / "hidden32" / "perf.csv", 32, 2),
        ("segr_h128", args.variant_root / "hidden128" / "debug", args.variant_root / "hidden128" / "perf.csv", 128, 2),
    ]
    rows: list[dict[str, Any]] = []
    circuits: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    for method, debug_dir, perf_path, _hidden, _layers in variants:
        perf = read_perf(perf_path)
        for circuit in args.circuits:
            item = context[circuit]
            ranked, chosen_family = debug_rank(debug_dir, circuit)
            detail, circuit_summary = eval_rank(
                circuit,
                method,
                ranked,
                item["static_order"],
                item["counts"],
                item["data"].node_names,
                args,
            )
            for row in detail:
                row["chosen_family"] = chosen_family
            circuit_summary["chosen_family"] = chosen_family
            rows.extend(detail)
            circuits.append(circuit_summary)
            p = perf.get(circuit, {})
            runtime_rows.append(
                make_runtime_row(
                    "architecture_ablation",
                    "epfl20",
                    args.vector_seed,
                    circuit,
                    method,
                    selector=p.get("total_seconds", 0.0),
                    total=p.get("total_seconds", 0.0),
                    note="selector runtime from the corresponding architecture run perf.csv; evaluation reuses selected-seed random-vector counts.",
                )
            )
    summary = summarize_circuit_seed(rows, circuits)
    manifest = {
        "seed": args.vector_seed,
        "vectors": args.vectors,
        "configurations": [
            {"method": METHOD_DEFAULT, "hidden": 64, "layers": 2},
            {"method": "segr_l1", "hidden": 64, "layers": 1},
            {"method": "segr_l3", "hidden": 64, "layers": 3},
            {"method": "segr_h32", "hidden": 32, "layers": 2},
            {"method": "segr_h128", "hidden": 128, "layers": 2},
        ],
        "oracle": "selected-seed random-vector stuck-at count oracle",
        "variant_root": str(args.variant_root),
    }
    write_rows(out_dir / "architecture_ablation_rows.csv", rows)
    write_rows(out_dir / "architecture_ablation_circuits.csv", circuits)
    write_rows(out_dir / "architecture_ablation_summary.csv", summary)
    write_json(out_dir / "run_manifest.json", manifest)
    write_rows(out_dir / "runtime_summary.csv", runtime_rows)
    write_method_report(
        out_dir / "architecture_ablation_report.md",
        "SEGR Architecture Ablation",
        summary,
        "Only layers or hidden dimension is changed; all evaluation rows use the same selected random-vector oracle.",
    )
    return runtime_rows


def read_debug_rows(debug_dir: Path, circuit: str) -> list[dict[str, str]]:
    return read_csv(debug_dir / f"{circuit}_node_debug.csv")


def rank_from_scores(rows: list[dict[str, str]], key: str) -> list[str]:
    return [
        row["node"]
        for row in sorted(
            rows,
            key=lambda r: (read_float(r, key), str(r.get("node", ""))),
            reverse=True,
        )
    ]


def no_gnn_rank(rows: list[dict[str, str]]) -> list[str]:
    scored = []
    for row in rows:
        score = (
            0.50 * read_float(row, "static_rank")
            + 0.35 * read_float(row, "cache_struct_rank")
            + 0.15 * read_float(row, "family_score")
        )
        scored.append((score, str(row["node"])))
    return [name for _, name in sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)]


def shuffled_gnn_rank(rows: list[dict[str, str]], seed: int, circuit: str) -> list[str]:
    names = [str(row["node"]) for row in rows]
    scores = np.asarray([read_float(row, "gnn_rank") for row in rows], dtype=float)
    rng = np.random.default_rng(seed + sum(ord(ch) for ch in circuit))
    shuffled = np.array(scores, copy=True)
    rng.shuffle(shuffled)
    return [
        name
        for _, name in sorted(zip(shuffled, names), key=lambda x: (float(x[0]), x[1]), reverse=True)
    ]


def structural_family_heuristic_rank(data: Any) -> list[str]:
    scored = []
    for name in data.node_names:
        feats = data.feature_by_name.get(name, {})
        score = (
            0.25 * float(feats.get("dist_min_inv", 0.0) or 0.0)
            + 0.25 * float(feats.get("dist_avg_inv", 0.0) or 0.0)
            + 0.20 * float(feats.get("reconv", 0.0) or 0.0)
            + 0.15 * float(feats.get("out_deg", 0.0) or 0.0)
            + 0.15 * float(feats.get("pagerank", 0.0) or 0.0)
        )
        scored.append((score, name))
    return [name for _, name in sorted(scored, key=lambda x: (x[0], x[1]), reverse=True)]


def materialize_component(args: argparse.Namespace, context: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = ensure_dir(args.analysis_root / f"v38_component_ablation_seed{args.vector_seed}_20260526_01")
    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    circuits: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    variants = [
        METHOD_DEFAULT,
        "no_gnn",
        "gnn_only",
        "shuffled_gnn",
        "pure_static_proximity",
        "cache_structural_only",
        "structural_family_only",
    ]
    perf = read_perf(args.default_perf_csv)
    for circuit in args.circuits:
        data = context[circuit]["data"]
        counts = context[circuit]["counts"]
        static_order = context[circuit]["static_order"]
        debug_rows = read_debug_rows(args.default_debug_dir, circuit)
        full_rank, chosen_family = debug_rank(args.default_debug_dir, circuit)
        ranking_by_variant = {
            METHOD_DEFAULT: full_rank,
            "no_gnn": no_gnn_rank(debug_rows),
            "gnn_only": rank_from_scores(debug_rows, "gnn_rank"),
            "shuffled_gnn": shuffled_gnn_rank(debug_rows, args.vector_seed, circuit),
            "pure_static_proximity": static_order,
            "cache_structural_only": rank_from_scores(debug_rows, "cache_struct_rank"),
            "structural_family_only": structural_family_heuristic_rank(data),
        }
        for variant in variants:
            detail, summary = eval_rank(circuit, variant, ranking_by_variant[variant], static_order, counts, data.node_names, args)
            for row in detail:
                row["chosen_family"] = chosen_family
            summary["chosen_family"] = chosen_family
            rows.extend(detail)
            circuits.append(summary)
            selector_seconds = perf.get(circuit, {}).get("total_seconds", 0.0) if variant == METHOD_DEFAULT else 0.0
            runtime_rows.append(
                make_runtime_row(
                    "component_ablation",
                    "epfl20",
                    args.vector_seed,
                    circuit,
                    variant,
                    selector=selector_seconds,
                    evaluation=0.0,
                    total=selector_seconds,
                    note="component ablation ranking derived from debug scores; offline counts are used only for evaluation.",
                )
            )
    summary = summarize_circuit_seed(rows, circuits)
    for row in runtime_rows:
        row["total_wall_seconds"] = time.perf_counter() - t0
    manifest = {
        "seed": args.vector_seed,
        "vectors": args.vectors,
        "variants": variants,
        "oracle": "selected-seed random-vector stuck-at count oracle",
        "notes": {
            "no_gnn": "0.50 static_rank + 0.35 cache_struct_rank + 0.15 family_score; no gnn_rank/final_score.",
            "shuffled_gnn": "gnn_rank values are deterministically shuffled within each circuit.",
            "structural_family_only": "fixed structural heuristic from distance, reconvergence, fanout, and PageRank features; no gnn_rank/final_score.",
        },
    }
    write_rows(out_dir / "component_ablation_rows.csv", rows)
    write_rows(out_dir / "component_ablation_circuits.csv", circuits)
    write_rows(out_dir / "component_ablation_summary.csv", summary)
    write_json(out_dir / "run_manifest.json", manifest)
    write_rows(out_dir / "runtime_summary.csv", runtime_rows)
    write_method_report(
        out_dir / "component_ablation_report.md",
        "SEGR Component / Method Ablation",
        summary,
        "All component variants use one global ranking per circuit and are evaluated under the selected seed7089 random-vector oracle.",
    )
    return runtime_rows


def materialize_fusa_fair(args: argparse.Namespace, context: dict[str, Any]) -> list[dict[str, Any]]:
    out_dir = ensure_dir(args.analysis_root / f"v38_fusa_supervised_fair_seed{args.vector_seed}_20260526_01")
    t0 = time.perf_counter()
    rows, circuits, summary, folds, runtime_rows = run_same_feature_loco(
        args,
        context,
        methods=["fusa_supervised_fair_mlp_loco"],
        experiment_name="fusa_supervised_fair",
    )
    wall = time.perf_counter() - t0
    for row in runtime_rows:
        row["total_wall_seconds"] = wall
    write_rows(out_dir / "fusa_rows.csv", rows)
    write_rows(out_dir / "fusa_circuits.csv", circuits)
    write_rows(out_dir / "fusa_method_summary.csv", summary)
    write_rows(out_dir / "fusa_folds.csv", folds)
    write_rows(out_dir / "runtime_summary.csv", runtime_rows)
    write_method_report(
        out_dir / "fusa_report.md",
        "FuSa / Supervised-Fair Baseline",
        summary,
        "This supervised-fair baseline uses the same runtime-visible SEGR features and LOCO protocol; held-out labels are used only for final evaluation.",
    )
    return runtime_rows


def materialize_iscas_from_existing(args: argparse.Namespace) -> list[dict[str, Any]]:
    """ISCAS85 evidence is generated by scripts/run_iscas85_89_main.py."""
    return []


def materialize_unified(args: argparse.Namespace, runtime_rows: list[dict[str, Any]]) -> None:
    out_dir = ensure_dir(args.analysis_root / f"v38_unified_reviewer_experiments_seed{args.vector_seed}_20260526_01")
    summary_sources = [
        ("main", args.analysis_root / f"v38_single_seed_main_{args.vector_seed}_20260526_01" / "main_seed_summary.csv"),
        ("same_feature", args.analysis_root / f"v38_same_feature_baselines_seed{args.vector_seed}_20260526_01" / "baseline_method_summary.csv"),
        ("architecture", args.analysis_root / f"v38_architecture_ablation_seed{args.vector_seed}_20260526_01" / "architecture_ablation_summary.csv"),
        ("component", args.analysis_root / f"v38_component_ablation_seed{args.vector_seed}_20260526_01" / "component_ablation_summary.csv"),
        ("fusa", args.analysis_root / f"v38_fusa_supervised_fair_seed{args.vector_seed}_20260526_01" / "fusa_method_summary.csv"),
    ]
    all_summary: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for experiment, path in summary_sources:
        exists = path.exists()
        manifest_rows.append(
            {
                "experiment": experiment,
                "path": str(path),
                "exists": exists,
                "seed": args.vector_seed,
                "vectors": args.vectors,
                "oracle": "selected-seed random-vector stuck-at counts" if experiment != "iscas85" else "ISCAS85 random-vector stuck-at counts",
            }
        )
        if exists:
            for row in read_csv(path):
                all_summary.append({"experiment": experiment, **row})
    write_rows(out_dir / "all_method_summary.csv", all_summary)
    write_rows(out_dir / "reproducibility_manifest.csv", manifest_rows)
    write_rows(out_dir / "runtime_summary_all.csv", runtime_rows)

    index_lines = [
        "# SEGR Unified Reviewer Experiment Index",
        "",
        f"Selected main seed: `{args.vector_seed}`; vectors: `{args.vectors}`.",
        "",
        "FI-JSON reconstruction is not used for the paper-facing SEGR main result.",
        "The SEGR unsupervised selector runtime does not read FI/oracle/random-vector counts; those counts are used only for offline evaluation and supervised training labels.",
        "",
        "| Experiment | Output | Exists |",
        "| --- | --- | ---: |",
    ]
    for row in manifest_rows:
        index_lines.append(f"| {row['experiment']} | `{row['path']}` | {row['exists']} |")
    (out_dir / "v38_unified_experiment_index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    five_seed = read_json(args.five_seed_dir / "vector_summary.json") if args.five_seed_dir.exists() else {}
    main_lines = [
        "# SEGR Main Summary",
        "",
        f"Main seed: `{args.vector_seed}`.",
        "",
        "| Method | Macro ideal | Loss rows | Random gate fails |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in all_summary:
        if row.get("experiment") == "main":
            main_lines.append(
                f"| {row.get('method', '')} | {float(row.get('macro_ideal_ratio_raw', 0.0)):.4f} | "
                f"{row.get('loss_rows', '')} | {row.get('random_gate_fail', '')} |"
            )
    (out_dir / "v38_main_summary.md").write_text("\n".join(main_lines) + "\n", encoding="utf-8")

    if five_seed:
        stability_lines = [
            "# SEGR 5-Seed Stability Summary",
            "",
            "This table is an independent random-vector stability archive. It is not used as a bridge source for the current main table.",
            "",
            "| Stability run | Seeds | Vectors | Macro mean | Macro std | Macro min | Macro max | Loss rows | Random gate fails |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| epfl20_5seed | {five_seed.get('seeds', '')} | {five_seed.get('vectors', '')} | "
                f"{float(five_seed.get('macro_mean_over_seeds', 0.0)):.4f} | "
                f"{float(five_seed.get('macro_std_over_seeds', 0.0)):.4f} | "
                f"{float(five_seed.get('macro_min_over_seeds', 0.0)):.4f} | "
                f"{float(five_seed.get('macro_max_over_seeds', 0.0)):.4f} | "
                f"{five_seed.get('loss_rows_total', '')} | {five_seed.get('random_gate_fail_total', '')} |"
            ),
        ]
        (out_dir / "v38_5seed_stability_summary.md").write_text("\n".join(stability_lines) + "\n", encoding="utf-8")

    response_lines = [
        "# SEGR Reviewer 2 Response Tables",
        "",
        "## Method Summary",
        "",
        "| Experiment | Method | Macro ideal | Loss rows | Random gate fails |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in all_summary:
        response_lines.append(
            f"| {row.get('experiment', '')} | {row.get('method', '')} | "
            f"{float(row.get('macro_ideal_ratio_raw', 0.0)):.4f} | {row.get('loss_rows', '')} | "
            f"{row.get('random_gate_fail', '')} |"
        )
    (out_dir / "v38_reviewer2_response_tables.md").write_text("\n".join(response_lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis"))
    parser.add_argument("--five-seed-dir", type=Path, default=Path("analysis/segr_epfl20_5seed_128vectors_20260526_01"))
    parser.add_argument("--count-cache-dir", type=Path, default=Path("analysis/v38_single_seed_7089_count_cache_20260526_01"))
    parser.add_argument("--default-debug-dir", type=Path, default=Path("outputs_runs/v38_no_hand_parameters_epfl20_20260526_01/debug"))
    parser.add_argument("--default-perf-csv", type=Path, default=Path("outputs_runs/v38_no_hand_parameters_epfl20_20260526_01/perf.csv"))
    parser.add_argument("--variant-root", type=Path, default=Path("outputs_runs/v38_architecture_ablation_20260526_01"))
    parser.add_argument("--circuits", nargs="+", default=EPFL20)
    parser.add_argument("--vectors", type=int, default=128)
    parser.add_argument("--vector-seed", type=int, default=7089)
    parser.add_argument("--engine", choices=["numba", "python"], default="numba")
    parser.add_argument("--count-mode", choices=["two_pass", "opposite_flip"], default="opposite_flip")
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=95289)
    parser.add_argument("--sample-per-circuit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1107)
    parser.add_argument("--lr-positive-quantile", type=float, default=0.80)
    parser.add_argument("--lr-max-iter", type=int, default=1000)
    parser.add_argument("--rf-trees", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=12)
    parser.add_argument("--rf-min-samples-leaf", type=int, default=2)
    parser.add_argument("--rf-jobs", type=int, default=-1)
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--rebuild-counts", action="store_true")
    args = parser.parse_args()

    runtime_rows: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    context, count_manifest, context_seconds = build_epfl_context(args)
    write_rows(args.analysis_root / f"v38_same_feature_baselines_seed{args.vector_seed}_20260526_01" / "count_manifest.csv", count_manifest)
    runtime_rows.extend(materialize_main_seed(args, context))
    runtime_rows.extend(materialize_baselines(args, context))
    runtime_rows.extend(materialize_architecture(args, context))
    runtime_rows.extend(materialize_component(args, context))
    runtime_rows.extend(materialize_fusa_fair(args, context))
    runtime_rows.extend(materialize_iscas_from_existing(args))
    runtime_rows.append(
        make_runtime_row(
            "reviewer_experiment_materialization",
            "epfl20",
            args.vector_seed,
            "ALL",
            "materialization",
            offline=context_seconds,
            total=time.perf_counter() - t_all,
            note="End-to-end wall time for materializing selected-seed reviewer experiment outputs.",
        )
    )
    materialize_unified(args, runtime_rows)
    print(
        json.dumps(
            {
                "seed": args.vector_seed,
                "vectors": args.vectors,
                "runtime_total_seconds": time.perf_counter() - t_all,
                "outputs": str(args.analysis_root / f"v38_unified_reviewer_experiments_seed{args.vector_seed}_20260526_01"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
