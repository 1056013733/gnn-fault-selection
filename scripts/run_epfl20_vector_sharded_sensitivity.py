#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from standalone.data_io import EPFL20, load_circuit  # noqa: E402


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


def shard_count_for(total: int, target_size: int, min_shards: int, max_shards: int) -> int:
    return max(int(min_shards), min(int(max_shards), int(math.ceil(total / max(1, target_size)))))


def expected_shard_summary(count_root: Path, circuit: str, vectors: int, seed: int, shard: int, shards: int) -> Path:
    return (
        count_root
        / circuit
        / f"vectors{int(vectors)}_seed{int(seed)}"
        / f"shard_{int(shard):04d}_of_{int(shards):04d}"
        / "count_shard_summary.json"
    )


def run_command(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError("command failed:\n" + " ".join(cmd) + "\n" + proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs_runs/v38_runtime_no_fi_epfl20_20260601_01/debug"))
    parser.add_argument("--count-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--circuits", nargs="+", default=EPFL20)
    parser.add_argument("--vectors", type=int, required=True)
    parser.add_argument("--vector-seed", type=int, default=7089)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--target-shard-size", type=int, default=4096)
    parser.add_argument("--min-shards", type=int, default=1)
    parser.add_argument("--max-shards", type=int, default=64)
    parser.add_argument("--count-mode", choices=["two_pass", "opposite_flip"], default="opposite_flip")
    parser.add_argument("--random-samples", type=int, default=1000)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    t_all = time.perf_counter()
    args.count_root.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_circuit_root = args.out_dir / "_per_circuit"
    plan_rows: list[dict[str, Any]] = []
    tasks: list[list[str]] = []

    for circuit in args.circuits:
        data = load_circuit(args.root, args.fi_root, circuit)
        total = len(data.node_names)
        shards = shard_count_for(total, int(args.target_shard_size), int(args.min_shards), int(args.max_shards))
        plan_rows.append({"circuit": circuit, "candidate_nodes": total, "shards": shards})
        for shard in range(shards):
            summary_path = expected_shard_summary(args.count_root, circuit, args.vectors, args.vector_seed, shard, shards)
            if summary_path.exists():
                continue
            tasks.append(
                [
                    sys.executable,
                    str(repo / "scripts" / "run_epfl20_vector_count_shard.py"),
                    "--root",
                    str(args.root),
                    "--fi-root",
                    str(args.fi_root),
                    "--count-root",
                    str(args.count_root),
                    "--circuit",
                    circuit,
                    "--vectors",
                    str(args.vectors),
                    "--vector-seed",
                    str(args.vector_seed),
                    "--shard-index",
                    str(shard),
                    "--shard-count",
                    str(shards),
                    "--count-mode",
                    args.count_mode,
                ]
            )

    write_rows(args.out_dir / "shard_plan.csv", plan_rows)
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = [pool.submit(run_command, cmd, repo) for cmd in tasks]
            for idx, fut in enumerate(as_completed(futures), start=1):
                fut.result()
                print(f"completed shard task {idx}/{len(futures)}", flush=True)

    merged_dirs: list[Path] = []
    for row in plan_rows:
        circuit = str(row["circuit"])
        merged = per_circuit_root / circuit
        cmd = [
            sys.executable,
            str(repo / "scripts" / "merge_epfl_vector_count_shards.py"),
            "--count-root",
            str(args.count_root),
            "--out-dir",
            str(merged),
            "--circuit",
            circuit,
            "--vectors",
            str(args.vectors),
            "--vector-seeds",
            str(args.vector_seed),
            "--expected-shard-count",
            str(row["shards"]),
            "--debug-dir",
            str(args.debug_dir),
            "--random-samples",
            str(args.random_samples),
        ]
        run_command(cmd, repo)
        merged_dirs.append(merged)

    aggregate_specs = [
        ("vector_rows.csv", "vector_rows.csv"),
        ("vector_circuit_seeds.csv", "vector_circuit_seeds.csv"),
        ("vector_circuits.csv", "vector_circuits.csv"),
        ("vector_manifest.csv", "vector_manifest.csv"),
    ]
    for source_name, out_name in aggregate_specs:
        rows: list[dict[str, str]] = []
        for merged in merged_dirs:
            rows.extend(read_csv(merged / source_name))
        write_rows(args.out_dir / out_name, rows)

    circuit_rows = read_csv(args.out_dir / "vector_circuits.csv")
    seed_rows = read_csv(args.out_dir / "vector_circuit_seeds.csv")
    macro = mean(float(row["seed_ratio_mean"]) for row in circuit_rows)
    seed_summary = [
        {
            "vector_seed": int(args.vector_seed),
            "circuits": len(seed_rows),
            "macro_ideal_ratio_raw": macro,
            "loss_rows": sum(int(float(row["loss_rows_total"])) for row in circuit_rows),
            "random_gate_fail": sum(int(float(row["random_gate_fail_total"])) for row in circuit_rows),
        }
    ]
    write_rows(args.out_dir / "vector_seed_summary.csv", seed_summary)
    summary = {
        "suite": "epfl20",
        "protocol": "epfl_gate_level_random_vector_stuck_at_candidate_sharded",
        "protocol_note": "Candidate-node shards are merged before oracle, static, SEGR, and random-prefix metrics are computed.",
        "circuits": len(circuit_rows),
        "seeds": 1,
        "vectors": int(args.vectors),
        "seed_eval_mode": "candidate_sharded",
        "engine": "numba",
        "count_mode": args.count_mode,
        "macro_mean_over_seeds": macro,
        "macro_std_over_seeds": 0.0,
        "macro_min_over_seeds": macro,
        "macro_max_over_seeds": macro,
        "loss_rows_total": seed_summary[0]["loss_rows"],
        "random_gate_fail_total": seed_summary[0]["random_gate_fail"],
        "runtime_total_seconds": time.perf_counter() - t_all,
        "workers": int(args.workers),
    }
    (args.out_dir / "vector_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
