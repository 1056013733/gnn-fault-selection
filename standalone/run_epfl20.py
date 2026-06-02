from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_io import EPFL20, FEATURE_KEYS, load_circuit
from .evaluate import detected_instance_counts, eval_selection, oracle_fault_instances, topk, write_rows
from .global_rank import build_global_rank
from .gnn_rank import score_gnn


SEGR_RANK_MODE = "segr_structure_derived_global_rank"


def static_ranked(data) -> list[str]:
    return sorted(
        data.node_names,
        key=lambda n: (float(data.static_score.get(n, 0.0) or 0.0), n),
        reverse=True,
    )


def prepare_x_without_semantics(x: np.ndarray) -> np.ndarray:
    out = np.asarray(x, dtype=np.float32).copy()
    if "name_len" in FEATURE_KEYS:
        out[:, FEATURE_KEYS.index("name_len")] = 0.0
    return out


def run_circuit(args: argparse.Namespace, circuit: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    t0 = time.perf_counter()
    data = load_circuit(args.root, args.fi_root, circuit, load_fi=not bool(args.runtime_no_fi))
    load_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    gnn = score_gnn(
        data.node_names,
        data.name_to_idx,
        prepare_x_without_semantics(data.x),
        data.edges,
        data.cache_struct_score,
        epochs=args.epochs,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        device=args.device,
        train_node_cap=args.train_node_cap,
        seed=args.seed,
    )
    gnn_seconds = time.perf_counter() - t1
    t2 = time.perf_counter()
    rank = build_global_rank(
        data.node_names,
        data.static_score,
        data.cache_struct_score,
        gnn.rank,
        gnn.diagnostics,
        feature_by_name=data.feature_by_name,
        name_to_idx=data.name_to_idx,
        edges=data.edges,
        x_np=data.x,
    )
    rank_seconds = time.perf_counter() - t2
    eval_data = data
    if bool(args.runtime_no_fi):
        eval_data = load_circuit(args.root, args.fi_root, circuit, load_fi=True)
        if eval_data.node_names != data.node_names:
            raise ValueError(f"{circuit}: runtime structural candidates differ from FI candidate set")
    counts = detected_instance_counts(eval_data.fi, data.node_names)
    static_order = static_ranked(data)
    rows: list[dict[str, Any]] = []
    for budget in args.budgets:
        k = max(1, min(len(data.node_names), int(__import__("math").ceil(float(budget) * len(data.node_names)))))
        oracle = oracle_fault_instances(counts, k)
        static_sel = topk(static_order, k)
        method_sel = topk(rank.ranked_nodes, k)
        static_eval = eval_selection(static_sel, counts, oracle)
        method_eval = eval_selection(method_sel, counts, oracle)
        common = {
            "circuit": circuit,
            "label_mode": "unsupervised",
            "budget": budget,
            "n_nodes": len(data.node_names),
            "budget_nodes": k,
            "rank_mode": SEGR_RANK_MODE,
            "gnn_nonrandom": rank.diagnostics.get("gnn_nonrandom", ""),
            "gnn_reliable": rank.diagnostics.get("gnn_reliable", ""),
            "gnn_struct_agree": rank.diagnostics.get("gnn_struct_agree", ""),
            "gnn_participated": float(rank.diagnostics.get("effective_gnn_weight", 0.0) or 0.0) > 0.0,
            "frontier_source": "structure_derived_rank_frontier",
            "global_rank_prefix_consistent": True,
            "runtime_no_fi": bool(args.runtime_no_fi),
            "chosen_family": rank.diagnostics.get("chosen_family", ""),
            "family_reason": rank.diagnostics.get("family_reason", ""),
            "family_peer_frontier_overlap": rank.diagnostics.get("family_peer_frontier_overlap", ""),
            "family_gnn_frontier_overlap": rank.diagnostics.get("family_gnn_frontier_overlap", ""),
            "family_cache_frontier_overlap": rank.diagnostics.get("family_cache_frontier_overlap", ""),
            "family_static_core_overlap": rank.diagnostics.get("family_static_core_overlap", ""),
        }
        rows.append({
            **common,
            "method": "pure_static_proximity",
            "rank_action": "static_proximity_global_prefix",
            "selected": len(static_sel),
            **static_eval,
        })
        rows.append({
            **common,
            "method": "segr_structure_derived_selector",
            "rank_action": "global_rank_prefix",
            "selected": len(method_sel),
            **method_eval,
        })
    perf = {
        "circuit": circuit,
        "load_seconds": load_seconds,
        "gnn_seconds": gnn_seconds,
        "rank_seconds": rank_seconds,
        "total_seconds": time.perf_counter() - t0,
        "runtime_no_fi": bool(args.runtime_no_fi),
        **rank.diagnostics,
    }
    for row in rank.node_debug:
        row["circuit"] = circuit
    return rows, perf, rank.node_debug


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--circuits", nargs="+", default=EPFL20)
    parser.add_argument("--budgets", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--perf", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-node-cap", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--runtime-no-fi", action="store_true")
    args = parser.parse_args()
    all_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    for circuit in args.circuits:
        rows, perf, node_debug = run_circuit(args, circuit)
        all_rows.extend(rows)
        perf_rows.append(perf)
        write_rows(args.debug_dir / f"{circuit}_node_debug.csv", node_debug)
    write_rows(args.output, all_rows)
    write_rows(args.perf, perf_rows)
    (args.debug_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(all_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
