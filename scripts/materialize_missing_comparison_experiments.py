#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from standalone.data_io import EPFL20, load_circuit, rank01  # noqa: E402
from standalone.evaluate import eval_selection, oracle_fault_instances, topk, write_rows  # noqa: E402
from standalone.structural_features import build_structural_families  # noqa: E402


METHOD_DEFAULT = "segr_structure_derived_selector"
BUDGETS = [0.05, 0.10, 0.20]
EPS = 1e-9
INF = 1.0e9
SUPPORTED_CELLS = {"AND2", "AND2B", "BUF", "INV", "NAND2", "NOR2", "OR2", "OR2B", "TIEHI", "TIELO"}


class Gate(NamedTuple):
    op: str
    out: str
    ins: tuple[str, ...]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_counts(path: Path) -> dict[str, int]:
    return {row["node"]: int(float(row.get("count", 0) or 0)) for row in read_csv(path)}


def normalize_name(token: str) -> str:
    item = token.strip()
    if item.startswith("\\"):
        item = item[1:].strip()
    if item.endswith("_fi_orig"):
        item = item[: -len("_fi_orig")]
    return item.strip()


def expand_bus_names(names: list[str], left: int, right: int) -> list[str]:
    lo = min(left, right)
    hi = max(left, right)
    out: list[str] = []
    for name in names:
        if "[" in name:
            out.append(name)
        else:
            out.extend(f"{name}[{idx}]" for idx in range(lo, hi + 1))
    return out


def split_decl_names(text: str) -> list[str]:
    text = re.sub(r"\b(?:wire|reg|logic|signed)\b", " ", text)
    bus = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", text)
    bus_range: tuple[int, int] | None = None
    if bus:
        bus_range = (int(bus.group(1)), int(bus.group(2)))
        text = f"{text[:bus.start()]} {text[bus.end():]}"
    out = [normalize_name(raw) for raw in text.split(",")]
    out = [name for name in out if name]
    if bus_range is not None:
        out = expand_bus_names(out, bus_range[0], bus_range[1])
    return out


def parse_ports(verilog: str, keyword: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(rf"\b{keyword}\b\s+([^;]+);", verilog, flags=re.S):
        names.extend(split_decl_names(match.group(1)))
    return list(dict.fromkeys(names))


def parse_gate_line(line: str) -> Gate | None:
    match = re.match(r"\s*([A-Za-z0-9]+)\s+[A-Za-z0-9_]+\s*\((.*)\)\s*;", line)
    if not match:
        return None
    op = match.group(1)
    if op not in SUPPORTED_CELLS:
        return None
    pins = {pin: normalize_name(value) for pin, value in re.findall(r"\.([A-Za-z0-9_]+)\((.*?)\)", match.group(2))}
    out = pins.get("Y")
    if not out:
        return None
    return Gate(op=op, out=out, ins=tuple(pins[p] for p in ("A", "B", "C", "D") if p in pins))


def parse_gate_level_verilog(path: Path) -> tuple[list[str], list[str], list[Gate]]:
    verilog = path.read_text(encoding="utf-8", errors="ignore")
    inputs = parse_ports(verilog, "input")
    outputs = parse_ports(verilog, "output")
    gates = [gate for line in verilog.splitlines() if (gate := parse_gate_line(line)) is not None]
    if not inputs or not outputs or not gates:
        raise ValueError(f"failed to parse gate-level netlist: {path}")
    return inputs, outputs, gates


def fi_verilog_path(root: Path, fi_root: str, circuit: str) -> Path:
    base = root / fi_root / circuit
    path = base / f"{circuit}_fi.v"
    if path.exists():
        return path
    matches = sorted(base.glob("*_fi.v"))
    if not matches:
        raise FileNotFoundError(f"missing *_fi.v for {circuit}")
    return matches[0]


def co_only_observability_scores(
    candidates: list[str],
    inputs: list[str],
    outputs: list[str],
    gates: list[Gate],
) -> dict[str, float]:
    cc0: dict[str, float] = {name: 1.0 for name in inputs}
    cc1: dict[str, float] = {name: 1.0 for name in inputs}
    for gate in gates:
        vals0 = [cc0.get(name, INF) for name in gate.ins]
        vals1 = [cc1.get(name, INF) for name in gate.ins]
        if gate.op == "TIEHI":
            cc0[gate.out], cc1[gate.out] = INF, 1.0
        elif gate.op == "TIELO":
            cc0[gate.out], cc1[gate.out] = 1.0, INF
        elif gate.op == "BUF":
            cc0[gate.out], cc1[gate.out] = vals0[0] + 1.0, vals1[0] + 1.0
        elif gate.op == "INV":
            cc0[gate.out], cc1[gate.out] = vals1[0] + 1.0, vals0[0] + 1.0
        elif gate.op == "AND2":
            cc0[gate.out], cc1[gate.out] = min(vals0) + 1.0, sum(vals1) + 1.0
        elif gate.op == "NAND2":
            cc0[gate.out], cc1[gate.out] = sum(vals1) + 1.0, min(vals0) + 1.0
        elif gate.op == "OR2":
            cc0[gate.out], cc1[gate.out] = sum(vals0) + 1.0, min(vals1) + 1.0
        elif gate.op == "NOR2":
            cc0[gate.out], cc1[gate.out] = min(vals1) + 1.0, sum(vals0) + 1.0
        elif gate.op == "AND2B":
            cc0[gate.out], cc1[gate.out] = min(vals0[0], vals1[1]) + 1.0, vals1[0] + vals0[1] + 1.0
        elif gate.op == "OR2B":
            cc0[gate.out], cc1[gate.out] = vals0[0] + vals1[1] + 1.0, min(vals1[0], vals0[1]) + 1.0

    co: dict[str, float] = defaultdict(lambda: INF)
    for name in outputs:
        co[name] = 0.0
    for gate in reversed(gates):
        out_co = co[gate.out]
        if out_co >= INF:
            continue
        vals0 = [cc0.get(name, INF) for name in gate.ins]
        vals1 = [cc1.get(name, INF) for name in gate.ins]
        for idx, name in enumerate(gate.ins):
            others = [j for j in range(len(gate.ins)) if j != idx]
            if gate.op in {"BUF", "INV"}:
                required = 0.0
            elif gate.op in {"AND2", "NAND2"}:
                required = sum(vals1[j] for j in others)
            elif gate.op in {"OR2", "NOR2"}:
                required = sum(vals0[j] for j in others)
            elif gate.op == "AND2B":
                required = vals0[1] if idx == 0 else vals1[0]
            elif gate.op == "OR2B":
                required = vals1[1] if idx == 0 else vals0[0]
            else:
                required = 0.0
            co[name] = min(co[name], out_co + required + 1.0)

    return {name: 1.0 / (1.0 + co[name]) if co[name] < INF else 0.0 for name in candidates}


def stable_static_order(data: Any) -> list[str]:
    return sorted(data.node_names, key=lambda n: (float(data.static_score.get(n, 0.0) or 0.0), n), reverse=True)


def row_ratio(method_value: float, oracle_value: float) -> float:
    return method_value / oracle_value if oracle_value else 1.0


def random_ratios(
    counts: dict[str, int],
    names: list[str],
    budgets: list[float],
    samples: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray([float(counts.get(name, 0) or 0) for name in names], dtype=float)
    out: list[float] = []
    n = len(names)
    for budget in budgets:
        k = max(1, min(n, math.ceil(float(budget) * n)))
        oracle = float(np.sum(np.sort(arr)[::-1][:k]))
        if oracle <= 0.0:
            out.append(1.0)
            continue
        for _ in range(int(samples)):
            idx = rng.choice(n, size=k, replace=False)
            out.append(float(np.sum(arr[idx])) / oracle)
    return np.asarray(out, dtype=float)


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
    del rows
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in circuit_rows:
        by_method.setdefault(str(row["method"]), []).append(row)
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
    rows = read_csv(debug_dir / f"{circuit}_node_debug.csv")
    ranked = [row["node"] for row in sorted(rows, key=lambda r: (float(r.get("global_rank", 0) or 0), r.get("node", "")))]
    chosen = str(rows[0].get("chosen_family", "")) if rows else ""
    return ranked, chosen


def build_epfl_context(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], float]:
    context: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for circuit in args.circuits:
        data = load_circuit(args.root, args.fi_root, circuit, load_fi=False)
        count_path = args.count_cache_dir / f"{circuit}_seed{args.vector_seed}_counts.csv"
        meta_path = args.count_cache_dir / f"{circuit}_seed{args.vector_seed}_meta.json"
        counts = load_counts(count_path)
        meta = read_json(meta_path) if meta_path.exists() else {}
        context[circuit] = {
            "data": data,
            "counts": counts,
            "static_order": stable_static_order(data),
        }
        manifest.append({"circuit": circuit, "count_path": str(count_path), **meta})
    return context, manifest, time.perf_counter() - t0


def score_rank(names: list[str], scores: dict[str, float]) -> list[str]:
    return sorted(names, key=lambda n: (float(scores.get(n, 0.0) or 0.0), n), reverse=True)


def average_rank_score(names: list[str], score_maps: list[dict[str, float]]) -> dict[str, float]:
    if not score_maps:
        return {name: 0.0 for name in names}
    ranked_maps = [rank01(scores, names) for scores in score_maps]
    return {
        name: float(mean(float(ranks.get(name, 0.0) or 0.0) for ranks in ranked_maps))
        for name in names
    }


def reciprocal_rank_fusion(names: list[str], rankings: list[list[str]], k: float = 60.0) -> dict[str, float]:
    out = {name: 0.0 for name in names}
    for ranking in rankings:
        for idx, name in enumerate(ranking):
            if name in out:
                out[name] += 1.0 / (k + idx + 1.0)
    return out


def family_scores(data: Any, debug_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    gnn_rank = {row["node"]: float(row.get("gnn_rank", 0.0) or 0.0) for row in debug_rows}
    final_score = {row["node"]: float(row.get("final_score", 0.0) or 0.0) for row in debug_rows}
    return build_structural_families(
        data.node_names,
        data.name_to_idx,
        data.edges,
        data.feature_by_name,
        data.cache_struct_score,
        gnn_rank,
        final_score,
    )


def build_rankings(data: Any, debug_rows: list[dict[str, str]]) -> dict[str, list[str]]:
    names = data.node_names
    families = family_scores(data, debug_rows)
    static_scores = data.static_score
    cache_scores = data.cache_struct_score
    gnn_scores = families["gnn_rank"]

    centrality_scores = average_rank_score(
        names,
        [
            families.get("pagerank", {}),
            families.get("betweenness", {}),
            families.get("eigen", {}),
        ],
    )
    observability_cone_scores = average_rank_score(
        names,
        [
            families.get("sink_near", {}),
            families.get("pdom_dist", {}),
            families.get("dist_pr", {}),
            families.get("sink_reach_near_pr", {}),
        ],
    )
    structural_borda_scores = average_rank_score(
        names,
        [
            static_scores,
            cache_scores,
            centrality_scores,
            observability_cone_scores,
        ],
    )
    equal_visible_fusion_scores = average_rank_score(names, [static_scores, cache_scores, gnn_scores])
    rrf_visible_scores = reciprocal_rank_fusion(
        names,
        [
            score_rank(names, static_scores),
            score_rank(names, cache_scores),
            score_rank(names, gnn_scores),
        ],
    )
    max_visible_scores = {
        name: max(
            float(static_scores.get(name, 0.0) or 0.0),
            float(cache_scores.get(name, 0.0) or 0.0),
            float(gnn_scores.get(name, 0.0) or 0.0),
        )
        for name in names
    }
    cone_centrality_scores = average_rank_score(
        names,
        [
            centrality_scores,
            observability_cone_scores,
            families.get("out_deg", {}),
        ],
    )

    score_by_method = {
        "centrality_only": centrality_scores,
        "observability_cone_proxy": observability_cone_scores,
        "cone_centrality_proxy": cone_centrality_scores,
        "structural_borda_no_gnn": structural_borda_scores,
        "equal_rank_fusion_static_cache_gnn": equal_visible_fusion_scores,
        "rrf_static_cache_gnn": rrf_visible_scores,
        "max_visible_signal": max_visible_scores,
    }
    return {method: score_rank(names, scores) for method, scores in score_by_method.items()}


def paired_stats(segr_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segr = {
        (str(row["circuit"]), float(row["budget"])): float(row["row_ideal_ratio_raw"])
        for row in segr_rows
    }
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in comparison_rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    out: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260602)
    ordered_methods = sorted(method for method in by_method if method != "co_only_observability")
    if "co_only_observability" in by_method:
        ordered_methods.append("co_only_observability")
    for method in ordered_methods:
        rows = by_method[method]
        deltas = []
        for row in rows:
            key = (str(row["circuit"]), float(row["budget"]))
            if key in segr:
                deltas.append(segr[key] - float(row["row_ideal_ratio_raw"]))
        arr = np.asarray(deltas, dtype=float)
        if arr.size:
            boot = np.asarray(
                [float(np.mean(rng.choice(arr, size=arr.size, replace=True))) for _ in range(10000)],
                dtype=float,
            )
            ci_lo, ci_hi = np.quantile(boot, [0.025, 0.975])
            positive = int(np.sum(arr > 0.0))
            negative = int(np.sum(arr < 0.0))
            zero = int(np.sum(arr == 0.0))
            out.append(
                {
                    "method": method,
                    "paired_rows": int(arr.size),
                    "mean_delta_segr_minus_method": float(np.mean(arr)),
                    "bootstrap_ci95_low": float(ci_lo),
                    "bootstrap_ci95_high": float(ci_hi),
                    "positive_rows": positive,
                    "negative_rows": negative,
                    "zero_rows": zero,
                }
            )
    return out


def write_report(path: Path, summary: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    lines = [
        "# Missing Comparison Experiments",
        "",
        "These rows supplement reviewer-requested comparison baselines without changing the SEGR ranking path.",
        "All methods produce one fixed target-circuit order before FI counts, RV-Oracle counts, random-vector outcomes, held-out labels, or evaluation metrics are opened.",
        "",
        "The observability rows are runtime-visible structural proxies rather than external tool measurements.",
        "",
        "## Method Summary",
        "",
        "| Method | Circuits | Macro ideal | Loss rows | Random gate fails | >=0.75 no-loss circuits |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['circuits']} | {float(row['macro_ideal_ratio_raw']):.4f} | "
            f"{row['loss_rows']} | {row['random_gate_fail']} | {row['ideal_ratio_075_circuits']} |"
        )
    lines.extend(
        [
            "",
            "## Paired SEGR Advantage",
            "",
            "| Method | Rows | Mean delta | 95% bootstrap CI | Positive | Negative | Zero |",
            "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in paired:
        lines.append(
            f"| {row['method']} | {row['paired_rows']} | "
            f"{float(row['mean_delta_segr_minus_method']):.4f} | "
            f"[{float(row['bootstrap_ci95_low']):.4f}, {float(row['bootstrap_ci95_high']):.4f}] | "
            f"{row['positive_rows']} | {row['negative_rows']} | {row['zero_rows']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize missing structural and fusion comparison baselines.")
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis"))
    parser.add_argument("--count-cache-dir", type=Path, default=Path("analysis/v38_single_seed_7089_count_cache_20260526_01"))
    parser.add_argument("--default-debug-dir", type=Path, default=Path("outputs_runs/v38_no_hand_parameters_epfl20_20260526_01/debug"))
    parser.add_argument("--circuits", nargs="+", default=EPFL20)
    parser.add_argument("--vectors", type=int, default=128)
    parser.add_argument("--vector-seed", type=int, default=7089)
    parser.add_argument("--engine", choices=["numba", "python"], default="numba")
    parser.add_argument("--count-mode", choices=["two_pass", "opposite_flip"], default="opposite_flip")
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=95289)
    parser.add_argument("--rebuild-counts", action="store_true")
    args = parser.parse_args()

    out_dir = ensure_dir(args.analysis_root / f"v38_missing_comparison_baselines_seed{args.vector_seed}_20260602_01")
    t0 = time.perf_counter()
    context, count_manifest, _ = build_epfl_context(args)

    rows: list[dict[str, Any]] = []
    circuits: list[dict[str, Any]] = []
    segr_rows: list[dict[str, Any]] = []
    for circuit in args.circuits:
        data = context[circuit]["data"]
        counts = context[circuit]["counts"]
        static_order = context[circuit]["static_order"]
        debug_rows = read_csv(args.default_debug_dir / f"{circuit}_node_debug.csv")
        segr_rank, chosen_family = debug_rank(args.default_debug_dir, circuit)
        detail, summary = eval_rank(circuit, METHOD_DEFAULT, segr_rank, static_order, counts, data.node_names, args)
        for row in detail:
            row["chosen_family"] = chosen_family
        segr_rows.extend(detail)

        rankings = build_rankings(data, debug_rows)
        inputs, outputs, gates = parse_gate_level_verilog(fi_verilog_path(args.root, args.fi_root, circuit))
        rankings["co_only_observability"] = score_rank(
            data.node_names,
            co_only_observability_scores(data.node_names, inputs, outputs, gates),
        )
        for method, ranking in rankings.items():
            detail, summary = eval_rank(circuit, method, ranking, static_order, counts, data.node_names, args)
            rows.extend(detail)
            circuits.append(summary)

    summary = summarize_circuit_seed(rows, circuits)
    paired = paired_stats(segr_rows, rows)
    manifest = {
        "seed": args.vector_seed,
        "vectors": args.vectors,
        "budgets": BUDGETS,
        "circuits": list(args.circuits),
        "elapsed_seconds": time.perf_counter() - t0,
        "source_debug_dir": str(args.default_debug_dir),
        "count_cache_dir": str(args.count_cache_dir),
        "forbidden_inputs": [
            "target FI labels before ranking",
            "target oracle counts before ranking",
            "target random-vector outcomes before ranking",
            "held-out labels before ranking",
            "baseline outcomes before ranking",
            "evaluation metrics before ranking",
        ],
        "method_definitions": {
            "co_only_observability": "output-observability-only ranking from the gate-level netlist; same cost propagation as the prior CO-only baseline, without controllability/fault-cost variants",
            "centrality_only": "average ranks of PageRank, betweenness, and eigenvector centrality features",
            "observability_cone_proxy": "runtime-visible proxy over sink-nearness, pdom-distance, distance-PageRank, and sink-reach-near PageRank features",
            "cone_centrality_proxy": "average ranks of centrality, observability-cone proxy, and out-degree",
            "structural_borda_no_gnn": "Borda-style average of static proximity, cache structure, centrality, and observability-cone proxy; no GNN",
            "equal_rank_fusion_static_cache_gnn": "equal rank average of static, cache, and GNN signals",
            "rrf_static_cache_gnn": "reciprocal-rank fusion over static, cache, and GNN rankings",
            "max_visible_signal": "per-node max over static, cache, and GNN rank scores",
        },
    }
    write_rows(out_dir / "comparison_rows.csv", rows)
    write_rows(out_dir / "comparison_circuits.csv", circuits)
    write_rows(out_dir / "comparison_method_summary.csv", summary)
    write_rows(out_dir / "comparison_paired_stats.csv", paired)
    write_rows(out_dir / "count_manifest.csv", count_manifest)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(out_dir / "comparison_report.md", summary, paired)
    print(json.dumps({"out_dir": str(out_dir), "methods": len(summary), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
