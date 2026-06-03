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
    method_ranked,
    parse_epfl_verilog,
    random_ratios,
    row_ratio,
)
from standalone.data_io import load_circuit  # noqa: E402
from standalone.evaluate import eval_selection, oracle_fault_instances, topk  # noqa: E402


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_shard_summaries(count_root: Path, circuit: str, vectors: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted(count_root.rglob("count_shard_summary.json")):
        row = read_json(path)
        if str(row.get("circuit")) != circuit:
            continue
        if int(row.get("vectors", -1)) != int(vectors):
            continue
        row["_dir"] = str(path.parent)
        out.append(row)
    return out


def validate_seed_shards(rows: list[dict[str, Any]], seed: int) -> tuple[int, int]:
    seed_rows = [row for row in rows if int(row["vector_seed"]) == int(seed)]
    if not seed_rows:
        raise ValueError(f"missing count shards for seed {seed}")
    shard_count = int(seed_rows[0]["shard_count"])
    candidate_total = int(seed_rows[0]["candidate_total"])
    partition_mode = str(seed_rows[0].get("partition_mode", "block") or "block")
    count_mode = str(seed_rows[0].get("count_mode", "two_pass") or "two_pass")
    seen: dict[int, dict[str, Any]] = {}
    for row in seed_rows:
        idx = int(row["shard_index"])
        if int(row["shard_count"]) != shard_count:
            raise ValueError(f"seed {seed}: inconsistent shard_count")
        if int(row["candidate_total"]) != candidate_total:
            raise ValueError(f"seed {seed}: inconsistent candidate_total")
        if str(row.get("partition_mode", "block") or "block") != partition_mode:
            raise ValueError(f"seed {seed}: inconsistent partition_mode")
        if str(row.get("count_mode", "two_pass") or "two_pass") != count_mode:
            raise ValueError(f"seed {seed}: inconsistent count_mode")
        seen[idx] = row
    missing = [idx for idx in range(shard_count) if idx not in seen]
    if missing:
        raise ValueError(f"seed {seed}: missing shard indexes {missing[:16]}")
    if partition_mode == "block":
        intervals = sorted(
            (int(row["candidate_start"]), int(row["candidate_stop"]), int(row["shard_index"]))
            for row in seen.values()
        )
        cursor = 0
        for start, stop, idx in intervals:
            if start != cursor:
                raise ValueError(f"seed {seed}: shard {idx} starts at {start}, expected {cursor}")
            if stop < start:
                raise ValueError(f"seed {seed}: shard {idx} has invalid interval {start}:{stop}")
            cursor = stop
        if cursor != candidate_total:
            raise ValueError(f"seed {seed}: shard coverage ends at {cursor}, expected {candidate_total}")
    elif partition_mode == "stride":
        covered = sum(int(row.get("candidate_position_count", 0) or 0) for row in seen.values())
        if covered != candidate_total:
            raise ValueError(f"seed {seed}: stride coverage has {covered} positions, expected {candidate_total}")
    else:
        raise ValueError(f"seed {seed}: unsupported partition_mode {partition_mode}")
    return shard_count, candidate_total


def load_counts_for_seed(rows: list[dict[str, Any]], seed: int) -> tuple[dict[str, int], float]:
    counts: dict[str, int] = {}
    seconds = 0.0
    for row in sorted(
        [item for item in rows if int(item["vector_seed"]) == int(seed)],
        key=lambda item: int(item["shard_index"]),
    ):
        seconds += float(row.get("seconds", 0.0) or 0.0)
        shard_csv = Path(str(row["_dir"])) / "fault_counts.csv"
        if not shard_csv.exists():
            raise FileNotFoundError(f"missing {shard_csv}")
        for item in read_csv(shard_csv):
            node = item["node"]
            value = int(float(item.get("count", 0) or 0))
            if value:
                if node in counts:
                    raise ValueError(f"duplicate node {node!r} in seed {seed}")
                counts[node] = value
    return counts, seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/runs/standalone_v06_iscas85_repair_epfl20_20260514_01/debug"))
    parser.add_argument("--count-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--vectors", type=int, default=128)
    parser.add_argument("--vector-seeds", nargs="+", type=int, default=[5089, 6089, 7089, 8089, 9089])
    parser.add_argument("--expected-shard-count", type=int, default=None)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=95289)
    args = parser.parse_args()

    t_all = time.perf_counter()
    data = load_circuit(args.root, args.fi_root, args.circuit)
    method_order, chosen_family = method_ranked(args.debug_dir, args.circuit)
    static_order = sorted(data.node_names, key=lambda n: (float(data.static_score.get(n, 0.0) or 0.0), n), reverse=True)
    fi_root = args.root / args.fi_root
    vcircuit = parse_epfl_verilog(fi_root, args.circuit, data.node_names)
    icircuit = compile_indexed(vcircuit)

    shard_summaries = load_shard_summaries(args.count_root, args.circuit, int(args.vectors))
    if args.expected_shard_count is not None:
        shard_summaries = [
            row for row in shard_summaries if int(row.get("shard_count", -1)) == int(args.expected_shard_count)
        ]
    if not shard_summaries:
        raise ValueError(f"no count shards found under {args.count_root} for {args.circuit}")

    row_out: list[dict[str, Any]] = []
    circuit_seed_out: list[dict[str, Any]] = []
    seed_ratios: list[float] = []
    seed_losses: list[int] = []
    seed_random_fails: list[int] = []
    seed_seconds: list[float] = []
    shard_counts: list[int] = []
    candidate_totals: list[int] = []

    for vector_seed in args.vector_seeds:
        shard_count, candidate_total = validate_seed_shards(shard_summaries, int(vector_seed))
        shard_counts.append(shard_count)
        candidate_totals.append(candidate_total)
        counts, count_seconds = load_counts_for_seed(shard_summaries, int(vector_seed))
        if candidate_total != len(data.node_names):
            raise ValueError(
                f"{args.circuit} seed {vector_seed}: candidate_total={candidate_total}, "
                f"data nodes={len(data.node_names)}"
            )
        ratios: list[float] = []
        losses = 0
        t_seed_eval = time.perf_counter()
        for budget in BUDGETS:
            k = max(1, min(len(data.node_names), math.ceil(float(budget) * len(data.node_names))))
            oracle = oracle_fault_instances(counts, k)
            static_eval = eval_selection(topk(static_order, k), counts, oracle)
            method_eval = eval_selection(topk(method_order, k), counts, oracle)
            static_value = float(static_eval["fault_instance_selected"])
            method_value = float(method_eval["fault_instance_selected"])
            loss = method_value < static_value - EPS
            ratio = row_ratio(method_value, float(oracle))
            losses += int(loss)
            ratios.append(ratio)
            row_out.append({
                "circuit": args.circuit,
                "vector_seed": int(vector_seed),
                "vectors": int(args.vectors),
                "budget": budget,
                "method_value": method_value,
                "static_value": static_value,
                "oracle_value": float(oracle),
                "row_ideal_ratio_raw": ratio,
                "loss": loss,
                "chosen_family": chosen_family,
            })
        rand_vals = random_ratios(counts, data.node_names, BUDGETS, int(args.random_samples), int(args.random_seed))
        random_mean = float(np.mean(rand_vals))
        random_p05 = float(np.quantile(rand_vals, 0.05))
        random_p95 = float(np.quantile(rand_vals, 0.95))
        gate = max(0.50, random_mean)
        method_ratio = float(mean(ratios))
        random_passed = method_ratio + 1e-12 >= gate
        seconds = count_seconds + (time.perf_counter() - t_seed_eval)
        seed_ratios.append(method_ratio)
        seed_losses.append(losses)
        seed_random_fails.append(0 if random_passed else 1)
        seed_seconds.append(seconds)
        circuit_seed_out.append({
            "circuit": args.circuit,
            "vector_seed": int(vector_seed),
            "vectors": int(args.vectors),
            "method_ratio": method_ratio,
            "loss_rows": losses,
            "random_mean": random_mean,
            "random_p05": random_p05,
            "random_p95": random_p95,
            "gate": gate,
            "random_passed": random_passed,
            "seconds": seconds,
            "chosen_family": chosen_family,
            "seed_eval_mode": "candidate_sharded",
            "engine": "numba",
            "shard_count": shard_count,
        })

    fault_seconds = sum(seed_seconds)
    circuit_out = [{
        "circuit": args.circuit,
        "chosen_family": chosen_family,
        "seed_ratio_mean": mean(seed_ratios),
        "seed_ratio_std": pstdev(seed_ratios) if len(seed_ratios) > 1 else 0.0,
        "seed_ratio_min": min(seed_ratios),
        "seed_ratio_max": max(seed_ratios),
        "loss_rows_total": sum(seed_losses),
        "random_gate_fail_total": sum(seed_random_fails),
        "seconds": time.perf_counter() - t_all,
        "seed_seconds_mean": mean(seed_seconds),
        "batched_fault_seconds": fault_seconds,
        "seed_eval_mode": "candidate_sharded",
        "engine": "numba",
        "inputs": len(vcircuit.inputs),
        "outputs": len(vcircuit.outputs),
        "gates": len(vcircuit.gates),
        "nets": icircuit.net_count,
        "candidate_nodes": len(data.node_names),
        "candidate_coverage": vcircuit.candidate_coverage,
        "unresolved_wire_count": len(vcircuit.unresolved_wires),
        "observed_output_count": vcircuit.observed_outputs,
        "shard_count": max(shard_counts) if shard_counts else "",
    }]
    manifest = [{
        "circuit": args.circuit,
        "inputs": len(vcircuit.inputs),
        "outputs": len(vcircuit.outputs),
        "gates": len(vcircuit.gates),
        "nets": icircuit.net_count,
        "candidate_nodes": len(data.node_names),
        "candidate_coverage": vcircuit.candidate_coverage,
        "unresolved_wire_count": len(vcircuit.unresolved_wires),
        "observed_output_count": vcircuit.observed_outputs,
        "batched_fault_seconds": fault_seconds,
        "seed_eval_mode": "candidate_sharded",
        "engine": "numba",
        "chosen_family": chosen_family,
        "protocol": "epfl_gate_level_random_vector_stuck_at_candidate_sharded",
        "shard_count": max(shard_counts) if shard_counts else "",
        "candidate_total": max(candidate_totals) if candidate_totals else "",
    }]
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in circuit_seed_out:
        by_seed[int(row["vector_seed"])].append(row)
    seed_summary: list[dict[str, Any]] = []
    for seed, items in sorted(by_seed.items()):
        seed_summary.append({
            "vector_seed": seed,
            "circuits": len(items),
            "macro_ideal_ratio_raw": mean(float(x["method_ratio"]) for x in items),
            "loss_rows": sum(int(x["loss_rows"]) for x in items),
            "random_gate_fail": sum(1 for x in items if not bool(x["random_passed"])),
        })
    macro_vals = [float(x["macro_ideal_ratio_raw"]) for x in seed_summary]
    external_wall_seconds: float | None = None
    for runtime_name in ("worker_scheduler_summary.json", "scheduler_summary.json"):
        runtime_path = args.count_root / runtime_name
        if runtime_path.exists():
            try:
                external_wall_seconds = float(read_json(runtime_path).get("wall_seconds", 0.0) or 0.0)
                break
            except Exception:
                external_wall_seconds = None
    summary = {
        "suite": "epfl20",
        "protocol": "epfl_gate_level_random_vector_stuck_at_candidate_sharded",
        "protocol_note": (
            "Candidate-level shards are mathematically equivalent to one seed-level run: each shard evaluates a "
            "disjoint node interval under the same PI vector seed, then fault counts are merged before oracle, "
            "static, method, and random-prefix metrics are computed."
        ),
        "circuits": 1,
        "seeds": len(args.vector_seeds),
        "vectors": int(args.vectors),
        "seed_eval_mode": "candidate_sharded",
        "engine": "numba",
        "macro_mean_over_seeds": mean(macro_vals) if macro_vals else 0.0,
        "macro_std_over_seeds": pstdev(macro_vals) if len(macro_vals) > 1 else 0.0,
        "macro_min_over_seeds": min(macro_vals) if macro_vals else 0.0,
        "macro_max_over_seeds": max(macro_vals) if macro_vals else 0.0,
        "loss_rows_total": sum(int(x["loss_rows"]) for x in seed_summary),
        "random_gate_fail_total": sum(int(x["random_gate_fail"]) for x in seed_summary),
        "runtime_total_seconds": external_wall_seconds if external_wall_seconds is not None else time.perf_counter() - t_all,
        "merge_runtime_seconds": time.perf_counter() - t_all,
        "fault_shard_cpu_seconds": fault_seconds,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "vector_rows.csv", row_out)
    write_rows(args.out_dir / "vector_circuit_seeds.csv", circuit_seed_out)
    write_rows(args.out_dir / "vector_circuits.csv", circuit_out)
    write_rows(args.out_dir / "vector_seed_summary.csv", seed_summary)
    write_rows(args.out_dir / "vector_manifest.csv", manifest)
    (args.out_dir / "vector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
