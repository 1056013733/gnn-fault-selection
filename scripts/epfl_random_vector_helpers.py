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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_epfl20_vector_stability import (  # noqa: E402
    BUDGETS,
    EPS,
    compile_indexed,
    compile_numba,
    method_ranked,
    parse_epfl_verilog,
    random_ratios,
    row_ratio,
    simulate_fault_counts_indexed,
    simulate_fault_counts_numba,
    simulate_fault_counts_numba_opposite,
)
from standalone.data_io import FEATURE_KEYS, rank01  # noqa: E402
from standalone.evaluate import eval_selection, oracle_fault_instances, topk, write_rows  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_counts(path: Path) -> dict[str, int]:
    return {row["node"]: int(float(row.get("count", 0) or 0)) for row in read_csv(path)}


def save_counts(path: Path, counts: dict[str, int]) -> None:
    write_rows(path, [{"node": node, "count": int(count)} for node, count in sorted(counts.items())])


def compute_or_load_counts(args: argparse.Namespace, circuit: str, data: Any) -> tuple[dict[str, int], dict[str, Any]]:
    count_path = args.count_cache_dir / f"{circuit}_seed{args.vector_seed}_counts.csv"
    meta_path = args.count_cache_dir / f"{circuit}_seed{args.vector_seed}_meta.json"
    if count_path.exists() and not args.rebuild_counts:
        counts = load_counts(count_path)
        meta = read_json(meta_path) if meta_path.exists() else {}
        meta["cache_hit"] = True
        return counts, meta

    t0 = time.perf_counter()
    fi_root = args.root / args.fi_root
    vcircuit = parse_epfl_verilog(fi_root, circuit, data.node_names)
    icircuit = compile_indexed(vcircuit)
    effective_seed = int(args.vector_seed) + sum(ord(ch) for ch in circuit)
    if args.engine == "numba":
        ncircuit = compile_numba(icircuit)
        if args.count_mode == "opposite_flip":
            counts = simulate_fault_counts_numba_opposite(ncircuit, int(args.vectors), effective_seed)
        else:
            counts = simulate_fault_counts_numba(ncircuit, int(args.vectors), effective_seed)
    else:
        counts = simulate_fault_counts_indexed(icircuit, int(args.vectors), effective_seed)
    seconds = time.perf_counter() - t0
    args.count_cache_dir.mkdir(parents=True, exist_ok=True)
    save_counts(count_path, counts)
    meta = {
        "circuit": circuit,
        "vector_seed": int(args.vector_seed),
        "effective_seed": effective_seed,
        "vectors": int(args.vectors),
        "engine": args.engine,
        "count_mode": args.count_mode,
        "seconds": seconds,
        "inputs": len(vcircuit.inputs),
        "outputs": len(vcircuit.outputs),
        "gates": len(vcircuit.gates),
        "candidate_nodes": len(data.node_names),
        "candidate_coverage": vcircuit.candidate_coverage,
        "cache_hit": False,
    }
    write_json(meta_path, meta)
    return counts, meta


def stable_static_order(data: Any) -> list[str]:
    return sorted(data.node_names, key=lambda n: (float(data.static_score.get(n, 0.0) or 0.0), n), reverse=True)


def eval_rank(
    circuit: str,
    method: str,
    ranked_nodes: list[str],
    static_order: list[str],
    counts: dict[str, int],
    names: list[str],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    losses = 0
    for budget in BUDGETS:
        k = max(1, min(len(names), math.ceil(float(budget) * len(names))))
        oracle = oracle_fault_instances(counts, k)
        static_eval = eval_selection(topk(static_order, k), counts, oracle)
        method_eval = eval_selection(topk(ranked_nodes, k), counts, oracle)
        static_value = float(static_eval["fault_instance_selected"])
        method_value = float(method_eval["fault_instance_selected"])
        loss = method_value < static_value - EPS
        ratio = row_ratio(method_value, float(oracle))
        losses += int(loss)
        ratios.append(ratio)
        rows.append(
            {
                "circuit": circuit,
                "vector_seed": int(args.vector_seed),
                "vectors": int(args.vectors),
                "budget": budget,
                "method": method,
                "method_value": method_value,
                "static_value": static_value,
                "oracle_value": float(oracle),
                "row_ideal_ratio_raw": ratio,
                "loss": loss,
            }
        )
    rand_vals = random_ratios(counts, names, BUDGETS, int(args.random_samples), int(args.random_seed))
    random_mean = float(np.mean(rand_vals))
    gate = max(0.50, random_mean)
    method_ratio = float(mean(ratios))
    return rows, {
        "circuit": circuit,
        "vector_seed": int(args.vector_seed),
        "vectors": int(args.vectors),
        "method": method,
        "method_ratio": method_ratio,
        "loss_rows": losses,
        "random_mean": random_mean,
        "random_p05": float(np.quantile(rand_vals, 0.05)),
        "random_p95": float(np.quantile(rand_vals, 0.95)),
        "gate": gate,
        "random_passed": method_ratio + 1e-12 >= gate,
    }


def summarize_circuit_seed(rows: list[dict[str, Any]], circuit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in circuit_rows:
        by_method[str(row["method"])].append(row)
    out: list[dict[str, Any]] = []
    for method, items in sorted(by_method.items()):
        vals = [float(x["method_ratio"]) for x in items]
        out.append(
            {
                "method": method,
                "circuits": len(items),
                "macro_ideal_ratio_raw": mean(vals) if vals else 0.0,
                "macro_std_over_circuits": pstdev(vals) if len(vals) > 1 else 0.0,
                "loss_rows": sum(int(x["loss_rows"]) for x in items),
                "random_gate_fail": sum(1 for x in items if not bool(x["random_passed"])),
                "ideal_ratio_075_circuits": sum(
                    1
                    for x in items
                    if float(x["method_ratio"]) >= 0.75 and int(x["loss_rows"]) == 0
                ),
            }
        )
    return out


def debug_rank(debug_dir: Path, circuit: str) -> tuple[list[str], str]:
    return method_ranked(debug_dir, circuit)


def read_debug_scores(debug_dir: Path, circuit: str) -> tuple[dict[str, float], dict[str, float]]:
    rows = read_csv(debug_dir / f"{circuit}_node_debug.csv")
    gnn = {row["node"]: float(row.get("gnn_rank", 0.0) or 0.0) for row in rows}
    final = {row["node"]: float(row.get("final_score", 0.0) or 0.0) for row in rows}
    return gnn, final


def make_feature_item(args: argparse.Namespace, circuit: str, data: Any, counts: dict[str, int]) -> dict[str, Any]:
    gnn_rank, final_score = read_debug_scores(args.default_debug_dir, circuit)
    count_rank = rank01(counts, data.node_names)
    idx = np.asarray([data.name_to_idx[name] for name in data.node_names], dtype=np.int64)
    feature_idx = [i for i, key in enumerate(FEATURE_KEYS) if key != "name_len"]
    base_x = data.x[idx[:, None], feature_idx].reshape(len(data.node_names), len(feature_idx)).astype(np.float32, copy=False)
    static = np.asarray([float(data.static_score.get(name, 0.0) or 0.0) for name in data.node_names], dtype=np.float32)
    cache = np.asarray([float(data.cache_struct_score.get(name, 0.0) or 0.0) for name in data.node_names], dtype=np.float32)
    gnn = np.asarray([float(gnn_rank.get(name, 0.0) or 0.0) for name in data.node_names], dtype=np.float32)
    final = np.asarray([float(final_score.get(name, 0.0) or 0.0) for name in data.node_names], dtype=np.float32)
    x = np.column_stack([base_x, static, cache, gnn, final]).astype(np.float32, copy=False)
    y_rank = np.asarray([float(count_rank.get(name, 0.0) or 0.0) for name in data.node_names], dtype=np.float32)
    count_arr = np.asarray([float(counts.get(name, 0) or 0) for name in data.node_names], dtype=np.float64)
    return {
        "x": x,
        "y_rank": y_rank,
        "counts": count_arr,
        "static": static.astype(np.float64),
        "names": np.asarray(data.node_names, dtype=str),
    }


def sample_indices(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False).astype(np.int64))


def positive_labels(y_rank: np.ndarray, quantile: float) -> np.ndarray:
    threshold = float(np.quantile(y_rank, quantile))
    labels = (y_rank >= threshold).astype(np.int32)
    if int(labels.sum()) == 0:
        labels[int(np.argmax(y_rank))] = 1
    if int(labels.sum()) == int(labels.size):
        labels[int(np.argmin(y_rank))] = 0
    return labels


def eval_array_predictions(
    circuit: str,
    method: str,
    pred: np.ndarray,
    item: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counts = item["counts"]
    static = item["static"]
    order = np.argsort(-pred, kind="stable")
    static_order = np.argsort(-static, kind="stable")
    oracle_order = np.argsort(-counts, kind="stable")
    rows: list[dict[str, Any]] = []
    ratios: list[float] = []
    losses = 0
    n = int(counts.size)
    for budget in BUDGETS:
        k = max(1, min(n, math.ceil(float(budget) * n)))
        method_value = float(np.sum(counts[order[:k]]))
        static_value = float(np.sum(counts[static_order[:k]]))
        oracle_value = float(np.sum(counts[oracle_order[:k]]))
        ratio = method_value / oracle_value if oracle_value else 1.0
        loss = method_value < static_value - EPS
        losses += int(loss)
        ratios.append(ratio)
        rows.append(
            {
                "circuit": circuit,
                "vector_seed": int(args.vector_seed),
                "vectors": int(args.vectors),
                "budget": budget,
                "method": method,
                "method_value": method_value,
                "static_value": static_value,
                "oracle_value": oracle_value,
                "row_ideal_ratio_raw": ratio,
                "loss": loss,
            }
        )
    return rows, {
        "circuit": circuit,
        "vector_seed": int(args.vector_seed),
        "vectors": int(args.vectors),
        "method": method,
        "method_ratio": float(mean(ratios)),
        "loss_rows": losses,
        "random_mean": "",
        "random_p05": "",
        "random_p95": "",
        "gate": "",
        "random_passed": "",
    }
