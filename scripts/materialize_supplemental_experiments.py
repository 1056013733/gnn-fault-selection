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
from statistics import mean, median, pstdev
from typing import Any, NamedTuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from standalone.data_io import EPFL20, load_circuit  # noqa: E402
from standalone.evaluate import eval_selection, oracle_fault_instances, topk, write_rows  # noqa: E402
from standalone.global_rank import CANDIDATE_FAMILIES, stable_order  # noqa: E402
from standalone.structural_features import build_structural_families  # noqa: E402


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


def gate_input_requirements(gate: Gate, target_pos: int) -> tuple[float, ...]:
    del target_pos
    return tuple(0.0 for _ in gate.ins)


def scoap_testability_scores(candidates: list[str], inputs: list[str], outputs: list[str], gates: list[Gate]) -> dict[str, float]:
    return scoap_testability_score_variants(candidates, inputs, outputs, gates)["scoap_min_fault_cost"]


def scoap_testability_score_variants(
    candidates: list[str],
    inputs: list[str],
    outputs: list[str],
    gates: list[Gate],
) -> dict[str, dict[str, float]]:
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

    co_only: dict[str, float] = {}
    min_fault: dict[str, float] = {}
    avg_fault: dict[str, float] = {}
    worst_fault: dict[str, float] = {}
    for name in candidates:
        observe_cost = co[name]
        sa0_cost = cc1.get(name, INF) + observe_cost
        sa1_cost = cc0.get(name, INF) + observe_cost
        min_cost = min(sa0_cost, sa1_cost)
        avg_cost = 0.5 * (sa0_cost + sa1_cost)
        worst_cost = max(sa0_cost, sa1_cost)
        co_only[name] = 1.0 / (1.0 + observe_cost) if observe_cost < INF else 0.0
        min_fault[name] = 1.0 / (1.0 + min_cost) if min_cost < INF else 0.0
        avg_fault[name] = 1.0 / (1.0 + avg_cost) if avg_cost < INF else 0.0
        worst_fault[name] = 1.0 / (1.0 + worst_cost) if worst_cost < INF else 0.0
    return {
        "scoap_co_only": co_only,
        "scoap_min_fault_cost": min_fault,
        "scoap_avg_fault_cost": avg_fault,
        "scoap_worst_fault_cost": worst_fault,
    }


def stable_static_order(data: Any) -> list[str]:
    return sorted(data.node_names, key=lambda n: (float(data.static_score.get(n, 0.0) or 0.0), n), reverse=True)


def row_ratio(method_value: float, oracle_value: float) -> float:
    return method_value / oracle_value if oracle_value else 1.0


def random_ratios(counts: dict[str, int], names: list[str], budgets: list[float], samples: int, seed: int) -> np.ndarray:
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
    random_gate_stats: dict[str, float] | None = None,
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
    if random_gate_stats is None:
        rand_vals = random_ratios(counts, names, BUDGETS, int(args.random_samples), int(args.random_seed))
        random_gate_stats = {
            "random_mean": float(np.mean(rand_vals)),
            "random_p05": float(np.quantile(rand_vals, 0.05)),
            "random_p95": float(np.quantile(rand_vals, 0.95)),
            "gate": max(0.50, float(np.mean(rand_vals))),
        }
    random_mean = float(random_gate_stats["random_mean"])
    gate = float(random_gate_stats["gate"])
    return rows, {
        "circuit": circuit,
        "vector_seed": int(args.vector_seed),
        "vectors": int(args.vectors),
        "method": method,
        "method_ratio": float(mean(ratios)),
        "loss_rows": losses,
        "random_mean": random_mean,
        "random_p05": float(random_gate_stats["random_p05"]),
        "random_p95": float(random_gate_stats["random_p95"]),
        "gate": gate,
        "random_passed": float(mean(ratios)) + 1e-12 >= gate,
    }


def summarize_circuit_seed(circuit_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                    1 for x in items if float(x["method_ratio"]) >= 0.75 and int(x["loss_rows"]) == 0
                ),
            }
        )
    return out


def build_family_scores(data: Any, debug_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    families = build_structural_families(
        data.node_names,
        data.name_to_idx,
        data.edges,
        data.feature_by_name,
        data.cache_struct_score,
        {row["node"]: float(row.get("gnn_rank", 0.0) or 0.0) for row in debug_rows},
        {row["node"]: float(row.get("final_score", 0.0) or 0.0) for row in debug_rows},
    )
    families["static_proximity"] = data.static_score
    return families


def debug_rank(debug_dir: Path, circuit: str) -> tuple[list[str], str, str]:
    rows = read_csv(debug_dir / f"{circuit}_node_debug.csv")
    ranked = [row["node"] for row in sorted(rows, key=lambda r: (float(r.get("global_rank", 0) or 0), r.get("node", "")))]
    chosen = str(rows[0].get("chosen_family", "")) if rows else ""
    reason = str(rows[0].get("family_reason", "")) if rows else ""
    return ranked, chosen, reason


def fi_verilog_path(root: Path, fi_root: str, circuit: str) -> Path:
    base = root / fi_root / circuit
    path = base / f"{circuit}_fi.v"
    if path.exists():
        return path
    matches = sorted(base.glob("*_fi.v"))
    if not matches:
        raise FileNotFoundError(f"missing *_fi.v for {circuit}")
    return matches[0]


def write_report(path: Path, summary: list[dict[str, Any]], selector_summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Supplemental SCOAP and Selector Diagnostics",
        "",
        "These supplemental rows do not change or retune SEGR. Ranking orders are fixed before offline FI counts, RV-Oracle counts, random-vector outcomes, held-out labels, baseline outcomes, or evaluation metrics are opened.",
        "",
        "The selector-family sweep is diagnostic only: it evaluates fixed runtime-visible candidate rankings after the rankings have been emitted, and it must not be used to select or retune SEGR.",
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
            "## Selector Family Diagnostics",
            "",
            "| Circuit | Chosen family | Rank among families | Chosen ratio | Best family | Best ratio | Median family ratio |",
            "| --- | --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for row in selector_summary:
        lines.append(
            f"| {row['circuit']} | {row['chosen_family']} | {row['chosen_family_rank']} | "
            f"{float(row['chosen_family_ratio']):.4f} | {row['best_family']} | "
            f"{float(row['best_family_ratio']):.4f} | {float(row['median_family_ratio']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize supplemental standard SCOAP-style and selector-family diagnostics.")
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--analysis-root", type=Path, default=Path("analysis"))
    parser.add_argument("--count-cache-dir", type=Path, default=Path("analysis/v38_single_seed_7089_count_cache_20260526_01"))
    parser.add_argument("--default-debug-dir", type=Path, default=Path("outputs_runs/v38_no_hand_parameters_epfl20_20260526_01/debug"))
    parser.add_argument("--circuits", nargs="+", default=EPFL20)
    parser.add_argument("--vectors", type=int, default=128)
    parser.add_argument("--vector-seed", type=int, default=7089)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=95289)
    args = parser.parse_args()

    out_dir = ensure_dir(args.analysis_root / f"v38_supplemental_selector_scoap_seed{args.vector_seed}_20260602_01")
    t0 = time.perf_counter()
    detail_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    selector_detail_rows: list[dict[str, Any]] = []
    selector_summary_rows: list[dict[str, Any]] = []
    count_manifest: list[dict[str, Any]] = []

    for circuit in args.circuits:
        data = load_circuit(args.root, args.fi_root, circuit, load_fi=False)
        counts = load_counts(args.count_cache_dir / f"{circuit}_seed{args.vector_seed}_counts.csv")
        meta_path = args.count_cache_dir / f"{circuit}_seed{args.vector_seed}_meta.json"
        count_manifest.append({"circuit": circuit, **(read_json(meta_path) if meta_path.exists() else {})})
        static_order = stable_static_order(data)
        debug_rows = read_csv(args.default_debug_dir / f"{circuit}_node_debug.csv")
        segr_rank, chosen_family, family_reason = debug_rank(args.default_debug_dir, circuit)
        rand_vals = random_ratios(counts, data.node_names, BUDGETS, int(args.random_samples), int(args.random_seed))
        random_gate_stats = {
            "random_mean": float(np.mean(rand_vals)),
            "random_p05": float(np.quantile(rand_vals, 0.05)),
            "random_p95": float(np.quantile(rand_vals, 0.95)),
            "gate": max(0.50, float(np.mean(rand_vals))),
        }

        inputs, outputs, gates = parse_gate_level_verilog(fi_verilog_path(args.root, args.fi_root, circuit))
        scoap_variants = scoap_testability_score_variants(data.node_names, inputs, outputs, gates)
        for method, scores in [
            ("standard_scoap_testability", scoap_variants["scoap_min_fault_cost"]),
            *scoap_variants.items(),
        ]:
            scoap_rank = stable_order(data.node_names, scores)
            rows, summary = eval_rank(
                circuit,
                method,
                scoap_rank,
                static_order,
                counts,
                data.node_names,
                args,
                random_gate_stats,
            )
            detail_rows.extend(rows)
            circuit_rows.append(summary)

        rows, summary = eval_rank(
            circuit,
            "segr_structure_derived_selector",
            segr_rank,
            static_order,
            counts,
            data.node_names,
            args,
            random_gate_stats,
        )
        detail_rows.extend(rows)
        circuit_rows.append(summary)

        family_scores = build_family_scores(data, debug_rows)
        family_results: list[dict[str, Any]] = []
        for family in CANDIDATE_FAMILIES:
            scores = family_scores.get(family)
            if not scores:
                continue
            family_rank = stable_order(data.node_names, scores)
            _rows, family_summary = eval_rank(
                circuit,
                f"selector_family_diagnostics:{family}",
                family_rank,
                static_order,
                counts,
                data.node_names,
                args,
                random_gate_stats,
            )
            family_summary["family"] = family
            family_results.append(family_summary)
            selector_detail_rows.extend(_rows)
        sorted_results = sorted(family_results, key=lambda row: float(row["method_ratio"]), reverse=True)
        ratios = [float(row["method_ratio"]) for row in sorted_results]
        chosen_ratio = next((float(row["method_ratio"]) for row in sorted_results if row["family"] == chosen_family), float("nan"))
        chosen_rank = next((idx + 1 for idx, row in enumerate(sorted_results) if row["family"] == chosen_family), -1)
        best = sorted_results[0] if sorted_results else {}
        selector_summary_rows.append(
            {
                "circuit": circuit,
                "chosen_family": chosen_family,
                "family_reason": family_reason,
                "chosen_family_rank": chosen_rank,
                "family_count": len(sorted_results),
                "chosen_family_ratio": chosen_ratio,
                "best_family": best.get("family", ""),
                "best_family_ratio": float(best.get("method_ratio", 0.0) or 0.0),
                "median_family_ratio": median(ratios) if ratios else 0.0,
                "diagnostic_only": True,
            }
        )

    method_summary = summarize_circuit_seed(circuit_rows)
    manifest = {
        "seed": args.vector_seed,
        "vectors": args.vectors,
        "budgets": BUDGETS,
        "elapsed_seconds": time.perf_counter() - t0,
        "source_debug_dir": str(args.default_debug_dir),
        "count_cache_dir": str(args.count_cache_dir),
        "experiments": [
            "standard_scoap_testability",
            "scoap_co_only",
            "scoap_min_fault_cost",
            "scoap_avg_fault_cost",
            "scoap_worst_fault_cost",
            "selector_family_diagnostics",
        ],
        "forbidden_inputs": [
            "target FI labels before ranking",
            "target oracle counts before ranking",
            "target random-vector outcomes before ranking",
            "held-out labels before ranking",
            "baseline outcomes before ranking",
            "evaluation metrics before ranking",
        ],
        "scoap_variant_note": "standard_scoap_testability is the optimistic min-fault-cost SCOAP-style row retained for compatibility; average/worst variants are stricter for aggregate polarity interpretation.",
        "selector_family_diagnostics_scope": "diagnostic only; post-ranking offline evaluation of fixed runtime-visible candidate families; not used to retune SEGR",
    }
    write_rows(out_dir / "supplemental_rows.csv", detail_rows)
    write_rows(out_dir / "supplemental_circuits.csv", circuit_rows)
    write_rows(out_dir / "supplemental_method_summary.csv", method_summary)
    write_rows(out_dir / "selector_family_rows.csv", selector_detail_rows)
    write_rows(out_dir / "selector_family_summary.csv", selector_summary_rows)
    write_rows(out_dir / "count_manifest.csv", count_manifest)
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(out_dir / "supplemental_report.md", method_summary, selector_summary_rows)
    print(json.dumps({"out_dir": str(out_dir), "methods": len(method_summary), "selector_circuits": len(selector_summary_rows)}, indent=2))


if __name__ == "__main__":
    main()
