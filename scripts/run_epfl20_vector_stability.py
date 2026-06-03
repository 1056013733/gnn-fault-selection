#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from standalone.data_io import EPFL20, load_circuit
from standalone.evaluate import eval_selection, oracle_fault_instances, topk, write_rows


BUDGETS = [0.05, 0.10, 0.20]
EPS = 1e-9
SUPPORTED_CELLS = {"AND2", "AND2B", "BUF", "INV", "NAND2", "NOR2", "OR2", "OR2B", "TIEHI", "TIELO"}
try:
    from numba import njit
except Exception:  # pragma: no cover
    njit = None


@dataclass(frozen=True)
class Gate:
    op: str
    out: str
    ins: tuple[str, ...]


@dataclass
class VerilogCircuit:
    circuit: str
    inputs: list[str]
    outputs: list[str]
    gates: list[Gate]
    candidates: list[str]
    candidate_coverage: int
    unresolved_wires: list[str]
    observed_outputs: int


@dataclass
class IndexedCircuit:
    circuit: str
    input_indices: list[int]
    output_indices: list[int]
    gates: list[tuple[int, int, tuple[int, ...]]]
    successors: list[list[int]]
    candidates: list[tuple[str, int]]
    net_count: int


@dataclass
class NumbaCircuit:
    circuit: str
    input_indices: np.ndarray
    gate_op: np.ndarray
    gate_out: np.ndarray
    gate_a: np.ndarray
    gate_b: np.ndarray
    succ_start: np.ndarray
    succ_edges: np.ndarray
    candidate_names: list[str]
    candidate_indices: np.ndarray
    observed: np.ndarray
    net_count: int


OP_CODE = {
    "TIEHI": 0,
    "TIELO": 1,
    "BUF": 2,
    "INV": 3,
    "AND2": 4,
    "NAND2": 5,
    "OR2": 6,
    "NOR2": 7,
    "AND2B": 8,
    "OR2B": 9,
}


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
            continue
        out.extend(f"{name}[{idx}]" for idx in range(lo, hi + 1))
    return out


def split_decl_names(text: str) -> list[str]:
    text = re.sub(r"\b(?:wire|reg|logic|signed)\b", " ", text)
    bus = re.search(r"\[\s*(\d+)\s*:\s*(\d+)\s*\]", text)
    bus_range: tuple[int, int] | None = None
    if bus:
        bus_range = (int(bus.group(1)), int(bus.group(2)))
        text = f"{text[:bus.start()]} {text[bus.end():]}"
    out: list[str] = []
    for raw in text.split(","):
        name = normalize_name(raw)
        if name:
            out.append(name)
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
    ins = tuple(pins[p] for p in ("A", "B", "C", "D") if p in pins)
    return Gate(op=op, out=out, ins=ins)


def parse_epfl_verilog(fi_root: Path, circuit: str, candidates: list[str]) -> VerilogCircuit:
    path = fi_root / circuit / f"{circuit}_fi.v"
    if not path.exists():
        matches = sorted((fi_root / circuit).glob("*_fi.v"))
        if not matches:
            raise FileNotFoundError(f"missing *_fi.v for {circuit}")
        path = matches[0]
    verilog = path.read_text(encoding="utf-8", errors="ignore")
    inputs = parse_ports(verilog, "input")
    outputs = parse_ports(verilog, "output")
    gates: list[Gate] = []
    unsupported: set[str] = set()
    for line in verilog.splitlines():
        inst = re.match(r"\s*([A-Za-z0-9]+)\s+[A-Za-z0-9_]+\s*\(", line)
        if inst and inst.group(1) not in SUPPORTED_CELLS:
            cell = inst.group(1)
            if cell not in {"module"}:
                unsupported.add(cell)
        gate = parse_gate_line(line)
        if gate is not None:
            gates.append(gate)
    if unsupported:
        raise ValueError(f"{circuit}: unsupported cells {sorted(unsupported)}")
    if not inputs or not outputs or not gates:
        raise ValueError(f"{circuit}: failed to parse inputs/outputs/gates from {path}")
    known_nets = set(inputs)
    known_nets.update(gate.out for gate in gates)
    observed_outputs = sum(1 for name in outputs if name in known_nets)
    candidate_coverage = sum(1 for name in candidates if name in known_nets)
    unresolved_wires = sorted(
        {
            inp
            for gate in gates
            for inp in gate.ins
            if const_value(inp, 1) is None and inp not in known_nets
        }
    )
    if unresolved_wires:
        sample = ", ".join(unresolved_wires[:10])
        raise ValueError(f"{circuit}: unresolved gate inputs ({len(unresolved_wires)}): {sample}")
    if observed_outputs != len(outputs):
        missing = sorted(name for name in outputs if name not in known_nets)
        sample = ", ".join(missing[:10])
        raise ValueError(f"{circuit}: unresolved observed outputs ({len(missing)}): {sample}")
    if candidate_coverage != len(candidates):
        missing = sorted(name for name in candidates if name not in known_nets)
        sample = ", ".join(missing[:10])
        raise ValueError(f"{circuit}: unresolved candidate nodes ({len(missing)}): {sample}")
    return VerilogCircuit(
        circuit=circuit,
        inputs=inputs,
        outputs=outputs,
        gates=gates,
        candidates=candidates,
        candidate_coverage=candidate_coverage,
        unresolved_wires=unresolved_wires,
        observed_outputs=observed_outputs,
    )


def const_value(token: str, vectors: int) -> np.ndarray | None:
    item = token.strip().lower()
    if item in {"1'b0", "1'h0", "0"}:
        return np.zeros(vectors, dtype=bool)
    if item in {"1'b1", "1'h1", "1"}:
        return np.ones(vectors, dtype=bool)
    return None


def eval_gate(op: str, ins: list[np.ndarray], vectors: int) -> np.ndarray:
    if op == "TIEHI":
        return np.ones(vectors, dtype=bool)
    if op == "TIELO":
        return np.zeros(vectors, dtype=bool)
    if op == "BUF":
        return ins[0]
    if op == "INV":
        return ~ins[0]
    if op == "AND2":
        return ins[0] & ins[1]
    if op == "NAND2":
        return ~(ins[0] & ins[1])
    if op == "OR2":
        return ins[0] | ins[1]
    if op == "NOR2":
        return ~(ins[0] | ins[1])
    if op == "AND2B":
        return ins[0] & (~ins[1])
    if op == "OR2B":
        return ins[0] | (~ins[1])
    raise ValueError(f"unsupported op {op}")


def const_mask(token: str, full_mask: int) -> int | None:
    item = token.strip().lower()
    if item in {"1'b0", "1'h0", "0"}:
        return 0
    if item in {"1'b1", "1'h1", "1"}:
        return full_mask
    return None


def eval_gate_mask(op: str, ins: list[int], full_mask: int) -> int:
    if op == "TIEHI":
        return full_mask
    if op == "TIELO":
        return 0
    if op == "BUF":
        return ins[0] & full_mask
    if op == "INV":
        return (~ins[0]) & full_mask
    if op == "AND2":
        return (ins[0] & ins[1]) & full_mask
    if op == "NAND2":
        return (~(ins[0] & ins[1])) & full_mask
    if op == "OR2":
        return (ins[0] | ins[1]) & full_mask
    if op == "NOR2":
        return (~(ins[0] | ins[1])) & full_mask
    if op == "AND2B":
        return (ins[0] & ((~ins[1]) & full_mask)) & full_mask
    if op == "OR2B":
        return (ins[0] | ((~ins[1]) & full_mask)) & full_mask
    raise ValueError(f"unsupported op {op}")


def eval_gate_code(code: int, ins: tuple[int, ...], full_mask: int) -> int:
    if code == 0:
        return full_mask
    if code == 1:
        return 0
    if code == 2:
        return ins[0] & full_mask
    if code == 3:
        return (~ins[0]) & full_mask
    if code == 4:
        return (ins[0] & ins[1]) & full_mask
    if code == 5:
        return (~(ins[0] & ins[1])) & full_mask
    if code == 6:
        return (ins[0] | ins[1]) & full_mask
    if code == 7:
        return (~(ins[0] | ins[1])) & full_mask
    if code == 8:
        return (ins[0] & ((~ins[1]) & full_mask)) & full_mask
    if code == 9:
        return (ins[0] | ((~ins[1]) & full_mask)) & full_mask
    raise ValueError(f"unsupported op code {code}")


def random_mask(rng: np.random.Generator, vectors: int) -> int:
    bits = rng.integers(0, 2, size=vectors, dtype=np.int8)
    value = 0
    for idx, bit in enumerate(bits):
        if int(bit):
            value |= 1 << idx
    return value


def require_mask(values: dict[str, int], name: str, circuit: str) -> int:
    if name not in values:
        raise ValueError(f"{circuit}: unresolved net during simulation: {name}")
    return values[name]


def golden_masks(circuit: VerilogCircuit, vectors: int, seed: int) -> tuple[dict[str, int], int]:
    rng = np.random.default_rng(seed)
    full_mask = (1 << vectors) - 1
    values: dict[str, int] = {}
    for name in circuit.inputs:
        values[name] = random_mask(rng, vectors)
    for gate in circuit.gates:
        ins: list[int] = []
        for inp in gate.ins:
            const = const_mask(inp, full_mask)
            if const is not None:
                ins.append(const)
            else:
                ins.append(require_mask(values, inp, circuit.circuit))
        values[gate.out] = eval_gate_mask(gate.op, ins, full_mask)
    return values, full_mask


def require_value(values: dict[str, np.ndarray], name: str, circuit: str) -> np.ndarray:
    if name not in values:
        raise ValueError(f"{circuit}: unresolved net during simulation: {name}")
    return values[name]


def golden_values(circuit: VerilogCircuit, vectors: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    values: dict[str, np.ndarray] = {}
    for name in circuit.inputs:
        values[name] = rng.integers(0, 2, size=vectors, dtype=np.int8).astype(bool)
    for gate in circuit.gates:
        ins: list[np.ndarray] = []
        for inp in gate.ins:
            const = const_value(inp, vectors)
            if const is not None:
                ins.append(const)
            else:
                ins.append(require_value(values, inp, circuit.circuit))
        values[gate.out] = eval_gate(gate.op, ins, vectors)
    return values


def build_successors(circuit: VerilogCircuit) -> dict[str, list[int]]:
    succs: dict[str, list[int]] = defaultdict(list)
    for gid, gate in enumerate(circuit.gates):
        for inp in gate.ins:
            if const_value(inp, 1) is None:
                succs[inp].append(gid)
    return succs


def compile_indexed(circuit: VerilogCircuit) -> IndexedCircuit:
    names: list[str] = []
    seen: set[str] = set()
    for name in circuit.inputs:
        if name not in seen:
            names.append(name)
            seen.add(name)
    for gate in circuit.gates:
        if gate.out not in seen:
            names.append(gate.out)
            seen.add(gate.out)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    gates: list[tuple[int, int, tuple[int, ...]]] = []
    successors: list[list[int]] = [[] for _ in names]
    for gid, gate in enumerate(circuit.gates):
        out_idx = name_to_idx[gate.out]
        in_idx = tuple(name_to_idx[inp] for inp in gate.ins if const_mask(inp, 1) is None)
        gates.append((OP_CODE[gate.op], out_idx, in_idx))
        for idx in in_idx:
            successors[idx].append(gid)
    return IndexedCircuit(
        circuit=circuit.circuit,
        input_indices=[name_to_idx[name] for name in circuit.inputs],
        output_indices=[name_to_idx[name] for name in circuit.outputs],
        gates=gates,
        successors=successors,
        candidates=[(name, name_to_idx[name]) for name in circuit.candidates],
        net_count=len(names),
    )


def compile_numba(circuit: IndexedCircuit) -> NumbaCircuit:
    gate_op = np.asarray([g[0] for g in circuit.gates], dtype=np.int32)
    gate_out = np.asarray([g[1] for g in circuit.gates], dtype=np.int32)
    gate_a = np.asarray([g[2][0] if len(g[2]) > 0 else -1 for g in circuit.gates], dtype=np.int32)
    gate_b = np.asarray([g[2][1] if len(g[2]) > 1 else -1 for g in circuit.gates], dtype=np.int32)
    counts = np.zeros(circuit.net_count, dtype=np.int32)
    for idx, succs in enumerate(circuit.successors):
        counts[idx] = len(succs)
    succ_start = np.zeros(circuit.net_count + 1, dtype=np.int32)
    np.cumsum(counts, out=succ_start[1:])
    succ_edges = np.empty(int(succ_start[-1]), dtype=np.int32)
    pos = 0
    for succs in circuit.successors:
        for gid in succs:
            succ_edges[pos] = gid
            pos += 1
    return NumbaCircuit(
        circuit=circuit.circuit,
        input_indices=np.asarray(circuit.input_indices, dtype=np.int32),
        gate_op=gate_op,
        gate_out=gate_out,
        gate_a=gate_a,
        gate_b=gate_b,
        succ_start=succ_start,
        succ_edges=succ_edges,
        candidate_names=[name for name, _ in circuit.candidates],
        candidate_indices=np.asarray([idx for _, idx in circuit.candidates], dtype=np.int32),
        observed=np.asarray([1 if idx in set(circuit.output_indices) else 0 for idx in range(circuit.net_count)], dtype=np.uint8),
        net_count=circuit.net_count,
    )


def input_masks_for_seed(input_count: int, vectors: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lo = np.zeros(input_count, dtype=np.uint64)
    hi = np.zeros(input_count, dtype=np.uint64)
    for pos in range(input_count):
        mask = random_mask(rng, vectors)
        lo[pos] = np.uint64(mask & ((1 << 64) - 1))
        hi[pos] = np.uint64((mask >> 64) & ((1 << 64) - 1))
    return lo, hi


def input_masks_for_seed_chunks(
    input_count: int,
    vectors: int,
    seed: int,
    chunk_size: int = 128,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    chunks = [
        (
            offset,
            min(int(chunk_size), int(vectors) - offset),
            np.zeros(input_count, dtype=np.uint64),
            np.zeros(input_count, dtype=np.uint64),
        )
        for offset in range(0, int(vectors), int(chunk_size))
    ]
    for pos in range(input_count):
        bits = rng.integers(0, 2, size=int(vectors), dtype=np.int8)
        for offset, width, lo, hi in chunks:
            lo_val = 0
            hi_val = 0
            for bit_index in range(width):
                if int(bits[offset + bit_index]):
                    if bit_index < 64:
                        lo_val |= 1 << bit_index
                    else:
                        hi_val |= 1 << (bit_index - 64)
            lo[pos] = np.uint64(lo_val)
            hi[pos] = np.uint64(hi_val)
    return [(width, lo, hi) for _offset, width, lo, hi in chunks]


if njit is not None:
    @njit(cache=True)
    def _popcount64(x: np.uint64) -> int:
        count = 0
        while x != 0:
            x = x & (x - np.uint64(1))
            count += 1
        return count


    @njit(cache=True)
    def _eval_gate_numba(op: int, alo: np.uint64, ahi: np.uint64, blo: np.uint64, bhi: np.uint64, full_lo: np.uint64, full_hi: np.uint64) -> tuple[np.uint64, np.uint64]:
        if op == 0:
            return full_lo, full_hi
        if op == 1:
            return np.uint64(0), np.uint64(0)
        if op == 2:
            return alo, ahi
        if op == 3:
            return (~alo) & full_lo, (~ahi) & full_hi
        if op == 4:
            return alo & blo, ahi & bhi
        if op == 5:
            return (~(alo & blo)) & full_lo, (~(ahi & bhi)) & full_hi
        if op == 6:
            return alo | blo, ahi | bhi
        if op == 7:
            return (~(alo | blo)) & full_lo, (~(ahi | bhi)) & full_hi
        if op == 8:
            return alo & ((~blo) & full_lo), ahi & ((~bhi) & full_hi)
        if op == 9:
            return alo | ((~blo) & full_lo), ahi | ((~bhi) & full_hi)
        return np.uint64(0), np.uint64(0)


    @njit(cache=True)
    def _simulate_counts_numba(
        vectors: int,
        input_indices: np.ndarray,
        input_lo: np.ndarray,
        input_hi: np.ndarray,
        gate_op: np.ndarray,
        gate_out: np.ndarray,
        gate_a: np.ndarray,
        gate_b: np.ndarray,
        succ_start: np.ndarray,
        succ_edges: np.ndarray,
        candidate_indices: np.ndarray,
        observed: np.ndarray,
        net_count: int,
    ) -> np.ndarray:
        if vectors >= 128:
            full_lo = np.uint64(0xFFFFFFFFFFFFFFFF)
            full_hi = np.uint64(0xFFFFFFFFFFFFFFFF)
        elif vectors > 64:
            full_lo = np.uint64(0xFFFFFFFFFFFFFFFF)
            full_hi = (np.uint64(1) << np.uint64(vectors - 64)) - np.uint64(1)
        elif vectors == 64:
            full_lo = np.uint64(0xFFFFFFFFFFFFFFFF)
            full_hi = np.uint64(0)
        else:
            full_lo = (np.uint64(1) << np.uint64(vectors)) - np.uint64(1)
            full_hi = np.uint64(0)

        gate_count = len(gate_op)
        base_lo = np.zeros(net_count, dtype=np.uint64)
        base_hi = np.zeros(net_count, dtype=np.uint64)
        for i in range(len(input_indices)):
            idx = input_indices[i]
            base_lo[idx] = input_lo[i]
            base_hi[idx] = input_hi[i]
        for gid in range(gate_count):
            a = gate_a[gid]
            b = gate_b[gid]
            alo = base_lo[a] if a >= 0 else np.uint64(0)
            ahi = base_hi[a] if a >= 0 else np.uint64(0)
            blo = base_lo[b] if b >= 0 else np.uint64(0)
            bhi = base_hi[b] if b >= 0 else np.uint64(0)
            lo, hi = _eval_gate_numba(gate_op[gid], alo, ahi, blo, bhi, full_lo, full_hi)
            out = gate_out[gid]
            base_lo[out] = lo
            base_hi[out] = hi

        counts = np.zeros(len(candidate_indices), dtype=np.int64)
        f_lo = np.zeros(net_count, dtype=np.uint64)
        f_hi = np.zeros(net_count, dtype=np.uint64)
        stamp = np.zeros(net_count, dtype=np.int32)
        qstamp = np.zeros(gate_count, dtype=np.int32)
        queue = np.empty(max(1, gate_count * 4), dtype=np.int32)
        sid = 1
        qid = 1
        for cpos in range(len(candidate_indices)):
            idx = candidate_indices[cpos]
            orig_lo = base_lo[idx]
            orig_hi = base_hi[idx]
            node_count = 0
            for stuck in range(2):
                stuck_lo = full_lo if stuck == 1 else np.uint64(0)
                stuck_hi = full_hi if stuck == 1 else np.uint64(0)
                if orig_lo == stuck_lo and orig_hi == stuck_hi:
                    continue
                sid += 1
                qid += 1
                if sid > 2100000000:
                    stamp[:] = 0
                    sid = 1
                if qid > 2100000000:
                    qstamp[:] = 0
                    qid = 1
                stamp[idx] = sid
                f_lo[idx] = stuck_lo
                f_hi[idx] = stuck_hi
                diff_lo = orig_lo ^ stuck_lo if observed[idx] != 0 else np.uint64(0)
                diff_hi = orig_hi ^ stuck_hi if observed[idx] != 0 else np.uint64(0)
                head = 0
                tail = 0
                for p in range(succ_start[idx], succ_start[idx + 1]):
                    gid = succ_edges[p]
                    if tail < len(queue):
                        queue[tail] = gid
                        tail += 1
                        qstamp[gid] = qid
                while head < tail:
                    gid = queue[head]
                    head += 1
                    qstamp[gid] = 0
                    a = gate_a[gid]
                    b = gate_b[gid]
                    if a >= 0 and stamp[a] == sid:
                        alo = f_lo[a]
                        ahi = f_hi[a]
                    elif a >= 0:
                        alo = base_lo[a]
                        ahi = base_hi[a]
                    else:
                        alo = np.uint64(0)
                        ahi = np.uint64(0)
                    if b >= 0 and stamp[b] == sid:
                        blo = f_lo[b]
                        bhi = f_hi[b]
                    elif b >= 0:
                        blo = base_lo[b]
                        bhi = base_hi[b]
                    else:
                        blo = np.uint64(0)
                        bhi = np.uint64(0)
                    new_lo, new_hi = _eval_gate_numba(gate_op[gid], alo, ahi, blo, bhi, full_lo, full_hi)
                    out = gate_out[gid]
                    if new_lo != base_lo[out] or new_hi != base_hi[out]:
                        if stamp[out] != sid or f_lo[out] != new_lo or f_hi[out] != new_hi:
                            stamp[out] = sid
                            f_lo[out] = new_lo
                            f_hi[out] = new_hi
                            if observed[out] != 0:
                                diff_lo = diff_lo | (base_lo[out] ^ new_lo)
                                diff_hi = diff_hi | (base_hi[out] ^ new_hi)
                            for p in range(succ_start[out], succ_start[out + 1]):
                                sgid = succ_edges[p]
                                if qstamp[sgid] != qid and tail < len(queue):
                                    queue[tail] = sgid
                                    tail += 1
                                    qstamp[sgid] = qid
                node_count += _popcount64(diff_lo) + _popcount64(diff_hi)
            counts[cpos] = node_count
        return counts


    @njit(cache=True)
    def _simulate_counts_numba_opposite(
        vectors: int,
        input_indices: np.ndarray,
        input_lo: np.ndarray,
        input_hi: np.ndarray,
        gate_op: np.ndarray,
        gate_out: np.ndarray,
        gate_a: np.ndarray,
        gate_b: np.ndarray,
        succ_start: np.ndarray,
        succ_edges: np.ndarray,
        candidate_indices: np.ndarray,
        observed: np.ndarray,
        net_count: int,
    ) -> np.ndarray:
        if vectors >= 128:
            full_lo = np.uint64(0xFFFFFFFFFFFFFFFF)
            full_hi = np.uint64(0xFFFFFFFFFFFFFFFF)
        elif vectors > 64:
            full_lo = np.uint64(0xFFFFFFFFFFFFFFFF)
            full_hi = (np.uint64(1) << np.uint64(vectors - 64)) - np.uint64(1)
        elif vectors == 64:
            full_lo = np.uint64(0xFFFFFFFFFFFFFFFF)
            full_hi = np.uint64(0)
        else:
            full_lo = (np.uint64(1) << np.uint64(vectors)) - np.uint64(1)
            full_hi = np.uint64(0)

        gate_count = len(gate_op)
        base_lo = np.zeros(net_count, dtype=np.uint64)
        base_hi = np.zeros(net_count, dtype=np.uint64)
        for i in range(len(input_indices)):
            idx = input_indices[i]
            base_lo[idx] = input_lo[i]
            base_hi[idx] = input_hi[i]
        for gid in range(gate_count):
            a = gate_a[gid]
            b = gate_b[gid]
            alo = base_lo[a] if a >= 0 else np.uint64(0)
            ahi = base_hi[a] if a >= 0 else np.uint64(0)
            blo = base_lo[b] if b >= 0 else np.uint64(0)
            bhi = base_hi[b] if b >= 0 else np.uint64(0)
            lo, hi = _eval_gate_numba(gate_op[gid], alo, ahi, blo, bhi, full_lo, full_hi)
            out = gate_out[gid]
            base_lo[out] = lo
            base_hi[out] = hi

        counts = np.zeros(len(candidate_indices), dtype=np.int64)
        f_lo = np.zeros(net_count, dtype=np.uint64)
        f_hi = np.zeros(net_count, dtype=np.uint64)
        stamp = np.zeros(net_count, dtype=np.int32)
        qstamp = np.zeros(gate_count, dtype=np.int32)
        queue = np.empty(max(1, gate_count * 4), dtype=np.int32)
        sid = 1
        qid = 1
        for cpos in range(len(candidate_indices)):
            idx = candidate_indices[cpos]
            orig_lo = base_lo[idx]
            orig_hi = base_hi[idx]
            fault_lo = (~orig_lo) & full_lo
            fault_hi = (~orig_hi) & full_hi
            sid += 1
            qid += 1
            if sid > 2100000000:
                stamp[:] = 0
                sid = 1
            if qid > 2100000000:
                qstamp[:] = 0
                qid = 1
            stamp[idx] = sid
            f_lo[idx] = fault_lo
            f_hi[idx] = fault_hi
            diff_lo = (orig_lo ^ fault_lo) if observed[idx] != 0 else np.uint64(0)
            diff_hi = (orig_hi ^ fault_hi) if observed[idx] != 0 else np.uint64(0)
            head = 0
            tail = 0
            for p in range(succ_start[idx], succ_start[idx + 1]):
                gid = succ_edges[p]
                if tail < len(queue):
                    queue[tail] = gid
                    tail += 1
                    qstamp[gid] = qid
            while head < tail:
                gid = queue[head]
                head += 1
                qstamp[gid] = 0
                a = gate_a[gid]
                b = gate_b[gid]
                if a >= 0 and stamp[a] == sid:
                    alo = f_lo[a]
                    ahi = f_hi[a]
                elif a >= 0:
                    alo = base_lo[a]
                    ahi = base_hi[a]
                else:
                    alo = np.uint64(0)
                    ahi = np.uint64(0)
                if b >= 0 and stamp[b] == sid:
                    blo = f_lo[b]
                    bhi = f_hi[b]
                elif b >= 0:
                    blo = base_lo[b]
                    bhi = base_hi[b]
                else:
                    blo = np.uint64(0)
                    bhi = np.uint64(0)
                new_lo, new_hi = _eval_gate_numba(gate_op[gid], alo, ahi, blo, bhi, full_lo, full_hi)
                out = gate_out[gid]
                if new_lo != base_lo[out] or new_hi != base_hi[out]:
                    if stamp[out] != sid or f_lo[out] != new_lo or f_hi[out] != new_hi:
                        stamp[out] = sid
                        f_lo[out] = new_lo
                        f_hi[out] = new_hi
                        if observed[out] != 0:
                            diff_lo = diff_lo | (base_lo[out] ^ new_lo)
                            diff_hi = diff_hi | (base_hi[out] ^ new_hi)
                        for p in range(succ_start[out], succ_start[out + 1]):
                            sgid = succ_edges[p]
                            if qstamp[sgid] != qid and tail < len(queue):
                                queue[tail] = sgid
                                tail += 1
                                qstamp[sgid] = qid
            counts[cpos] = _popcount64(diff_lo) + _popcount64(diff_hi)
        return counts


def simulate_fault_counts_numba(circuit: NumbaCircuit, vectors: int, seed: int) -> dict[str, int]:
    if njit is None:
        raise RuntimeError("numba is not available")
    if int(vectors) <= 128:
        input_lo, input_hi = input_masks_for_seed(len(circuit.input_indices), vectors, seed)
        counts = _simulate_counts_numba(
            int(vectors),
            circuit.input_indices,
            input_lo,
            input_hi,
            circuit.gate_op,
            circuit.gate_out,
            circuit.gate_a,
            circuit.gate_b,
            circuit.succ_start,
            circuit.succ_edges,
            circuit.candidate_indices,
            circuit.observed,
            int(circuit.net_count),
        )
    else:
        counts = np.zeros(len(circuit.candidate_indices), dtype=np.int64)
        for width, input_lo, input_hi in input_masks_for_seed_chunks(len(circuit.input_indices), vectors, seed):
            counts += _simulate_counts_numba(
                int(width),
                circuit.input_indices,
                input_lo,
                input_hi,
                circuit.gate_op,
                circuit.gate_out,
                circuit.gate_a,
                circuit.gate_b,
                circuit.succ_start,
                circuit.succ_edges,
                circuit.candidate_indices,
                circuit.observed,
                int(circuit.net_count),
            )
    return {name: int(counts[pos]) for pos, name in enumerate(circuit.candidate_names) if int(counts[pos]) != 0}


def simulate_fault_counts_numba_opposite(circuit: NumbaCircuit, vectors: int, seed: int) -> dict[str, int]:
    if njit is None:
        raise RuntimeError("numba is not available")
    if int(vectors) <= 128:
        input_lo, input_hi = input_masks_for_seed(len(circuit.input_indices), vectors, seed)
        counts = _simulate_counts_numba_opposite(
            int(vectors),
            circuit.input_indices,
            input_lo,
            input_hi,
            circuit.gate_op,
            circuit.gate_out,
            circuit.gate_a,
            circuit.gate_b,
            circuit.succ_start,
            circuit.succ_edges,
            circuit.candidate_indices,
            circuit.observed,
            int(circuit.net_count),
        )
    else:
        counts = np.zeros(len(circuit.candidate_indices), dtype=np.int64)
        for width, input_lo, input_hi in input_masks_for_seed_chunks(len(circuit.input_indices), vectors, seed):
            counts += _simulate_counts_numba_opposite(
                int(width),
                circuit.input_indices,
                input_lo,
                input_hi,
                circuit.gate_op,
                circuit.gate_out,
                circuit.gate_a,
                circuit.gate_b,
                circuit.succ_start,
                circuit.succ_edges,
                circuit.candidate_indices,
                circuit.observed,
                int(circuit.net_count),
            )
    return {name: int(counts[pos]) for pos, name in enumerate(circuit.candidate_names) if int(counts[pos]) != 0}


def golden_indexed(circuit: IndexedCircuit, vectors: int, seed: int) -> tuple[list[int], int]:
    rng = np.random.default_rng(seed)
    full_mask = (1 << vectors) - 1
    values = [0] * circuit.net_count
    for idx in circuit.input_indices:
        values[idx] = random_mask(rng, vectors)
    for code, out_idx, in_idx in circuit.gates:
        ins = tuple(values[idx] for idx in in_idx)
        values[out_idx] = eval_gate_code(code, ins, full_mask)
    return values, full_mask


def golden_indexed_multi(circuit: IndexedCircuit, vectors: int, seeds: list[int]) -> tuple[list[int], int]:
    lane_mask = (1 << vectors) - 1
    full_mask = (1 << (vectors * len(seeds))) - 1
    rngs = [np.random.default_rng(seed) for seed in seeds]
    values = [0] * circuit.net_count
    for idx in circuit.input_indices:
        combined = 0
        for lane, rng in enumerate(rngs):
            combined |= random_mask(rng, vectors) << (lane * vectors)
        values[idx] = combined
    for code, out_idx, in_idx in circuit.gates:
        ins = tuple(values[idx] for idx in in_idx)
        values[out_idx] = eval_gate_code(code, ins, full_mask)
    return values, full_mask


def simulate_fault_counts_indexed(circuit: IndexedCircuit, vectors: int, seed: int) -> dict[str, int]:
    values, full_mask = golden_indexed(circuit, vectors, seed)
    observed = [False] * circuit.net_count
    for idx in circuit.output_indices:
        observed[idx] = True
    counts: dict[str, int] = {}
    for name, idx in circuit.candidates:
        original = values[idx]
        node_count = 0
        for stuck in (False, True):
            stuck_val = full_mask if stuck else 0
            if original == stuck_val:
                continue
            affected: dict[int, int] = {idx: stuck_val}
            diff_obs = (original ^ stuck_val) if observed[idx] else 0
            queue = deque(circuit.successors[idx])
            queued = set(queue)
            while queue:
                gid = queue.popleft()
                queued.discard(gid)
                code, out_idx, in_idx = circuit.gates[gid]
                ins = tuple(affected.get(inp_idx, values[inp_idx]) for inp_idx in in_idx)
                new_val = eval_gate_code(code, ins, full_mask)
                base_val = values[out_idx]
                if new_val != base_val:
                    old_val = affected.get(out_idx)
                    if old_val is None or old_val != new_val:
                        affected[out_idx] = new_val
                        if observed[out_idx]:
                            diff_obs |= base_val ^ new_val
                        for succ_gid in circuit.successors[out_idx]:
                            if succ_gid not in queued:
                                queue.append(succ_gid)
                                queued.add(succ_gid)
            node_count += int(diff_obs.bit_count())
        if node_count:
            counts[name] = node_count
    return counts


def simulate_fault_counts_indexed_multi(circuit: IndexedCircuit, vectors: int, seeds: list[int]) -> list[dict[str, int]]:
    values, full_mask = golden_indexed_multi(circuit, vectors, seeds)
    lane_mask = (1 << vectors) - 1
    observed = [False] * circuit.net_count
    for idx in circuit.output_indices:
        observed[idx] = True
    counts_by_seed: list[dict[str, int]] = [dict() for _ in seeds]
    for name, idx in circuit.candidates:
        original = values[idx]
        node_counts = [0] * len(seeds)
        for stuck in (False, True):
            stuck_val = full_mask if stuck else 0
            if original == stuck_val:
                continue
            affected: dict[int, int] = {idx: stuck_val}
            diff_obs = (original ^ stuck_val) if observed[idx] else 0
            queue = deque(circuit.successors[idx])
            queued = set(queue)
            while queue:
                gid = queue.popleft()
                queued.discard(gid)
                code, out_idx, in_idx = circuit.gates[gid]
                ins = tuple(affected.get(inp_idx, values[inp_idx]) for inp_idx in in_idx)
                new_val = eval_gate_code(code, ins, full_mask)
                base_val = values[out_idx]
                if new_val != base_val:
                    old_val = affected.get(out_idx)
                    if old_val is None or old_val != new_val:
                        affected[out_idx] = new_val
                        if observed[out_idx]:
                            diff_obs |= base_val ^ new_val
                        for succ_gid in circuit.successors[out_idx]:
                            if succ_gid not in queued:
                                queue.append(succ_gid)
                                queued.add(succ_gid)
            if diff_obs:
                for lane in range(len(seeds)):
                    node_counts[lane] += int(((diff_obs >> (lane * vectors)) & lane_mask).bit_count())
        for lane, count in enumerate(node_counts):
            if count:
                counts_by_seed[lane][name] = count
    return counts_by_seed


def simulate_fault_counts(circuit: VerilogCircuit, vectors: int, seed: int) -> dict[str, int]:
    values, full_mask = golden_masks(circuit, vectors, seed)
    observed = [name for name in circuit.outputs if name in values]
    observed_set = set(observed)
    succs = build_successors(circuit)
    counts: dict[str, int] = {}
    for name in circuit.candidates:
        if name not in values:
            raise ValueError(f"{circuit.circuit}: candidate missing from simulation values: {name}")
        original = values[name]
        node_count = 0
        for stuck in (False, True):
            stuck_val = full_mask if stuck else 0
            if original == stuck_val:
                continue
            affected: dict[str, int] = {name: stuck_val}
            diff_obs = 0
            if name in observed_set:
                diff_obs |= original ^ stuck_val
            queue = deque(succs.get(name, []))
            queued = set(queue)
            while queue:
                gid = queue.popleft()
                queued.discard(gid)
                gate = circuit.gates[gid]
                ins: list[int] = []
                for inp in gate.ins:
                    const = const_mask(inp, full_mask)
                    if const is not None:
                        ins.append(const)
                    else:
                        ins.append(affected.get(inp, require_mask(values, inp, circuit.circuit)))
                new_val = eval_gate_mask(gate.op, ins, full_mask)
                base_val = require_mask(values, gate.out, circuit.circuit)
                if new_val != base_val:
                    old_val = affected.get(gate.out)
                    if old_val is None or old_val != new_val:
                        affected[gate.out] = new_val
                        if gate.out in observed_set:
                            diff_obs |= base_val ^ new_val
                        for succ_gid in succs.get(gate.out, []):
                            if succ_gid not in queued:
                                queue.append(succ_gid)
                                queued.add(succ_gid)
            node_count += int(diff_obs.bit_count())
        if node_count:
            counts[name] = node_count
    return counts


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def method_ranked(debug_dir: Path, circuit: str) -> tuple[list[str], str]:
    rows = read_csv(debug_dir / f"{circuit}_node_debug.csv")
    ranked = [row["node"] for row in sorted(rows, key=lambda r: int(float(r.get("global_rank", 10**12) or 10**12)))]
    family = rows[0].get("chosen_family", "") if rows else ""
    return ranked, family


def row_ratio(selected: float, oracle: float) -> float:
    return selected / oracle if oracle else 1.0


def random_ratios(counts: dict[str, int], names: list[str], budgets: list[float], samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(names)
    count_arr = np.asarray([float(counts.get(name, 0) or 0) for name in names], dtype=float)
    ks = [max(1, min(n, math.ceil(float(budget) * n))) for budget in budgets]
    kmax = max(ks)
    oracles = [oracle_fault_instances(counts, k) for k in ks]
    out: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(n, size=kmax, replace=False)
        prefix = np.cumsum(count_arr[sampled])
        vals = []
        for k, oracle in zip(ks, oracles):
            hit = float(prefix[k - 1])
            vals.append(row_ratio(hit, oracle))
        out.append(float(mean(vals)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/Supplementary_Experiments"))
    parser.add_argument("--fi-root", default="full_injection_results_verilator")
    parser.add_argument("--debug-dir", type=Path, default=Path("outputs/runs/standalone_v06_iscas85_repair_epfl20_20260514_01/debug"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--circuits", nargs="+", default=EPFL20)
    parser.add_argument("--vectors", type=int, default=128)
    parser.add_argument("--vector-seeds", nargs="+", type=int, default=[5089, 6089, 7089, 8089, 9089])
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=95289)
    parser.add_argument("--seed-eval-mode", choices=["separate", "batched"], default="separate")
    parser.add_argument("--engine", choices=["numba", "python"], default="numba")
    parser.add_argument("--count-mode", choices=["two_pass", "opposite_flip"], default="opposite_flip")
    args = parser.parse_args()
    if args.engine == "numba" and njit is None:
        raise RuntimeError("requested --engine numba but numba is not available")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fi_root = args.root / args.fi_root
    row_out: list[dict[str, Any]] = []
    circuit_seed_out: list[dict[str, Any]] = []
    circuit_out: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    t_all = time.perf_counter()
    for circuit_name in args.circuits:
        t_circuit = time.perf_counter()
        data = load_circuit(args.root, args.fi_root, circuit_name)
        method_order, chosen_family = method_ranked(args.debug_dir, circuit_name)
        static_order = sorted(data.node_names, key=lambda n: (float(data.static_score.get(n, 0.0) or 0.0), n), reverse=True)
        vcircuit = parse_epfl_verilog(fi_root, circuit_name, data.node_names)
        icircuit = compile_indexed(vcircuit)
        ncircuit = compile_numba(icircuit) if args.engine == "numba" else None
        seed_ratios: list[float] = []
        seed_losses: list[int] = []
        seed_random_fails: list[int] = []
        seed_seconds: list[float] = []
        t_fault = time.perf_counter()
        effective_seeds = [int(seed) + sum(ord(ch) for ch in circuit_name) for seed in args.vector_seeds]
        fault_seconds_by_seed: list[float] = []
        if args.seed_eval_mode == "batched":
            counts_by_seed = simulate_fault_counts_indexed_multi(icircuit, int(args.vectors), effective_seeds)
            batch_seconds = time.perf_counter() - t_fault
            fault_seconds_by_seed = [batch_seconds / max(1, len(args.vector_seeds)) for _ in args.vector_seeds]
        else:
            counts_by_seed = []
            for seed in effective_seeds:
                t_one = time.perf_counter()
                if args.engine == "numba":
                    assert ncircuit is not None
                    if args.count_mode == "two_pass":
                        counts_by_seed.append(simulate_fault_counts_numba(ncircuit, int(args.vectors), seed))
                    else:
                        counts_by_seed.append(simulate_fault_counts_numba_opposite(ncircuit, int(args.vectors), seed))
                else:
                    counts_by_seed.append(simulate_fault_counts_indexed(icircuit, int(args.vectors), seed))
                fault_seconds_by_seed.append(time.perf_counter() - t_one)
        fault_seconds = time.perf_counter() - t_fault
        for seed_pos, vector_seed in enumerate(args.vector_seeds):
            t_seed = time.perf_counter()
            counts = counts_by_seed[seed_pos]
            ratios: list[float] = []
            losses = 0
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
                    "circuit": circuit_name,
                    "vector_seed": vector_seed,
                    "vectors": args.vectors,
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
            seconds = fault_seconds_by_seed[seed_pos] + (time.perf_counter() - t_seed)
            seed_ratios.append(method_ratio)
            seed_losses.append(losses)
            seed_random_fails.append(0 if random_passed else 1)
            seed_seconds.append(seconds)
            circuit_seed_out.append({
                "circuit": circuit_name,
                "vector_seed": vector_seed,
                "vectors": args.vectors,
                "method_ratio": method_ratio,
                "loss_rows": losses,
                "random_mean": random_mean,
                "random_p05": random_p05,
                "random_p95": random_p95,
                "gate": gate,
                "random_passed": random_passed,
                "seconds": seconds,
                "chosen_family": chosen_family,
            })
        circuit_out.append({
            "circuit": circuit_name,
            "chosen_family": chosen_family,
            "seed_ratio_mean": mean(seed_ratios),
            "seed_ratio_std": pstdev(seed_ratios) if len(seed_ratios) > 1 else 0.0,
            "seed_ratio_min": min(seed_ratios),
            "seed_ratio_max": max(seed_ratios),
            "loss_rows_total": sum(seed_losses),
            "random_gate_fail_total": sum(seed_random_fails),
            "seconds": time.perf_counter() - t_circuit,
            "seed_seconds_mean": mean(seed_seconds),
            "batched_fault_seconds": fault_seconds,
            "seed_eval_mode": args.seed_eval_mode,
            "engine": args.engine,
            "count_mode": args.count_mode,
            "inputs": len(vcircuit.inputs),
            "outputs": len(vcircuit.outputs),
            "gates": len(vcircuit.gates),
            "nets": icircuit.net_count,
            "candidate_nodes": len(data.node_names),
            "candidate_coverage": vcircuit.candidate_coverage,
            "unresolved_wire_count": len(vcircuit.unresolved_wires),
            "observed_output_count": vcircuit.observed_outputs,
        })
        manifest.append({
            "circuit": circuit_name,
            "inputs": len(vcircuit.inputs),
            "outputs": len(vcircuit.outputs),
            "gates": len(vcircuit.gates),
            "nets": icircuit.net_count,
            "candidate_nodes": len(data.node_names),
            "candidate_coverage": vcircuit.candidate_coverage,
            "unresolved_wire_count": len(vcircuit.unresolved_wires),
            "observed_output_count": vcircuit.observed_outputs,
            "batched_fault_seconds": fault_seconds,
            "seed_eval_mode": args.seed_eval_mode,
            "engine": args.engine,
            "count_mode": args.count_mode,
            "chosen_family": chosen_family,
            "protocol": "epfl_gate_level_random_vector_stuck_at",
        })
        print(
            f"{circuit_name}: pi={len(vcircuit.inputs)} po={len(vcircuit.outputs)} "
            f"gates={len(vcircuit.gates)} coverage={vcircuit.candidate_coverage}/{len(data.node_names)} "
            f"unresolved={len(vcircuit.unresolved_wires)} mode={args.seed_eval_mode} engine={args.engine} mean={mean(seed_ratios):.6f} "
            f"std={pstdev(seed_ratios) if len(seed_ratios)>1 else 0.0:.6f} "
            f"loss={sum(seed_losses)} random_fail={sum(seed_random_fails)} "
            f"seconds={time.perf_counter() - t_circuit:.3f}",
            flush=True,
        )

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
    summary = {
        "suite": "epfl20",
        "protocol": "epfl_gate_level_random_vector_stuck_at",
        "protocol_note": (
            "Protocol matches the ISCAS'85 vector-stability check: for each vector seed, generate the same "
            "fixed number of random PI vectors, recompute stuck-at fault counts, keep one circuit-level global "
            "ranking fixed, and evaluate budget prefixes offline. The default seed mode runs repeated seeds "
            "separately; the optional batched mode packs seeds into independent bit lanes as a mathematical "
            "equivalence probe. FI/oracle are not runtime selector inputs."
        ),
        "circuits": len(args.circuits),
        "seeds": len(args.vector_seeds),
        "vectors": args.vectors,
        "seed_eval_mode": args.seed_eval_mode,
        "engine": args.engine,
        "count_mode": args.count_mode,
        "macro_mean_over_seeds": mean(macro_vals) if macro_vals else 0.0,
        "macro_std_over_seeds": pstdev(macro_vals) if len(macro_vals) > 1 else 0.0,
        "macro_min_over_seeds": min(macro_vals) if macro_vals else 0.0,
        "macro_max_over_seeds": max(macro_vals) if macro_vals else 0.0,
        "loss_rows_total": sum(int(x["loss_rows"]) for x in seed_summary),
        "random_gate_fail_total": sum(int(x["random_gate_fail"]) for x in seed_summary),
        "runtime_total_seconds": time.perf_counter() - t_all,
    }
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
