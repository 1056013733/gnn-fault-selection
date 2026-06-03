#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_epfl20_vector_stability import (  # noqa: E402
    NumbaCircuit,
    compile_indexed,
    compile_numba,
    parse_epfl_verilog,
    simulate_fault_counts_numba,
    simulate_fault_counts_numba_opposite,
)
from standalone.data_io import load_circuit  # noqa: E402


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


def subset_numba_circuit(circuit: NumbaCircuit, start: int, stop: int) -> NumbaCircuit:
    return replace(
        circuit,
        candidate_names=circuit.candidate_names[start:stop],
        candidate_indices=circuit.candidate_indices[start:stop],
    )


def block_interval(total: int, shard_index: int, shard_count: int) -> tuple[int, int]:
    start = (total * shard_index) // shard_count
    stop = (total * (shard_index + 1)) // shard_count
    return start, stop


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--count-root", type=Path, required=True)
    parser.add_argument("--circuit", required=True)
    parser.add_argument("--vectors", type=int, required=True)
    parser.add_argument("--vector-seed", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--count-mode", choices=["two_pass", "opposite_flip"], default="opposite_flip")
    args = parser.parse_args()

    t0 = time.perf_counter()
    data = load_circuit(args.root, args.fi_root, args.circuit)
    vcircuit = parse_epfl_verilog(args.root / args.fi_root, args.circuit, data.node_names)
    icircuit = compile_indexed(vcircuit)
    ncircuit = compile_numba(icircuit)
    total = len(ncircuit.candidate_names)
    start, stop = block_interval(total, int(args.shard_index), int(args.shard_count))
    shard = subset_numba_circuit(ncircuit, start, stop)
    effective_seed = int(args.vector_seed) + sum(ord(ch) for ch in args.circuit)
    if args.count_mode == "two_pass":
        counts = simulate_fault_counts_numba(shard, int(args.vectors), effective_seed)
    else:
        counts = simulate_fault_counts_numba_opposite(shard, int(args.vectors), effective_seed)
    seconds = time.perf_counter() - t0

    out_dir = (
        args.count_root
        / args.circuit
        / f"vectors{int(args.vectors)}_seed{int(args.vector_seed)}"
        / f"shard_{int(args.shard_index):04d}_of_{int(args.shard_count):04d}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(out_dir / "fault_counts.csv", [{"node": node, "count": count} for node, count in sorted(counts.items())])
    summary = {
        "circuit": args.circuit,
        "vectors": int(args.vectors),
        "vector_seed": int(args.vector_seed),
        "effective_seed": effective_seed,
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "candidate_start": start,
        "candidate_stop": stop,
        "candidate_position_count": stop - start,
        "candidate_total": total,
        "partition_mode": "block",
        "engine": "numba",
        "count_mode": args.count_mode,
        "seconds": seconds,
    }
    (out_dir / "count_shard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
