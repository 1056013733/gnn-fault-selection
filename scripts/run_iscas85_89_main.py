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
from statistics import mean
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from standalone.data_io import FEATURE_KEYS, adaptive_cache_struct_score, minmax, rank01
from standalone.evaluate import eval_selection, oracle_fault_instances, topk, write_rows
from standalone.global_rank import build_global_rank
from standalone.gnn_rank import score_gnn


ISCAS85 = ["c432", "c499", "c880", "c1355", "c1908", "c2670", "c3540", "c5315", "c6288", "c7552"]
ISCAS89 = ["s27", "s298", "s344", "s382", "s420.1", "s510", "s641", "s820", "s953", "s1196", "s1423", "s5378"]
BUDGETS = [0.05, 0.10, 0.20]


@dataclass(frozen=True)
class Gate:
    out: str
    op: str
    ins: tuple[str, ...]


@dataclass
class BenchCircuit:
    suite: str
    circuit: str
    path: Path
    inputs: list[str]
    outputs: list[str]
    dff_q: list[str]
    dff_d: list[str]
    gates: list[Gate]
    node_names: list[str]
    name_to_idx: dict[str, int]
    edges: list[tuple[int, int]]
    candidates: list[str]
    observed: list[str]
    topo_gates: list[Gate]
    x: np.ndarray
    feature_by_name: dict[str, dict[str, float]]
    static_score: dict[str, float]
    cache_struct_score: dict[str, float]
    stats: dict[str, Any]


def safe_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def is_const(name: str) -> bool:
    return name in {"0", "1", "1'b0", "1'b1", "1'bx", "1'bX"}


def const_value(name: str, n: int) -> np.ndarray:
    return np.ones(n, dtype=bool) if name in {"1", "1'b1"} else np.zeros(n, dtype=bool)


def parse_bench(path: Path, suite: str) -> tuple[list[str], list[str], list[str], list[str], list[Gate]]:
    inputs: list[str] = []
    outputs: list[str] = []
    dff_q: list[str] = []
    dff_d: list[str] = []
    gates: list[Gate] = []
    input_re = re.compile(r"INPUT\(([^)]+)\)", re.IGNORECASE)
    output_re = re.compile(r"OUTPUT\(([^)]+)\)", re.IGNORECASE)
    assign_re = re.compile(r"([^=\s]+)\s*=\s*([A-Za-z0-9_]+)\(([^)]*)\)")
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = input_re.fullmatch(line)
        if m:
            inputs.append(m.group(1).strip())
            continue
        m = output_re.fullmatch(line)
        if m:
            outputs.append(m.group(1).strip())
            continue
        m = assign_re.fullmatch(line)
        if not m:
            raise ValueError(f"unsupported bench line in {path}: {raw}")
        out = m.group(1).strip()
        op = m.group(2).strip().upper()
        ins = tuple(x.strip() for x in m.group(3).split(",") if x.strip())
        if op == "DFF":
            dff_q.append(out)
            if ins:
                dff_d.append(ins[0])
        else:
            gates.append(Gate(out=out, op=op, ins=ins))
    if suite == "iscas85":
        dff_q = []
        dff_d = []
    return inputs, outputs, dff_q, dff_d, gates


def topo_sort_gates(gates: list[Gate], sources: set[str]) -> list[Gate]:
    remaining = list(gates)
    known = set(sources)
    ordered: list[Gate] = []
    while remaining:
        next_remaining: list[Gate] = []
        progressed = False
        for gate in remaining:
            if all(is_const(inp) or inp in known for inp in gate.ins):
                ordered.append(gate)
                known.add(gate.out)
                progressed = True
            else:
                next_remaining.append(gate)
        if not progressed:
            ordered.extend(next_remaining)
            break
        remaining = next_remaining
    return ordered


def eval_gate(op: str, ins: list[np.ndarray]) -> np.ndarray:
    if op == "NOT":
        return np.logical_not(ins[0])
    if op in {"BUFF", "BUF"}:
        return ins[0].copy()
    if op == "AND":
        out = ins[0].copy()
        for arr in ins[1:]:
            out &= arr
        return out
    if op == "NAND":
        return np.logical_not(eval_gate("AND", ins))
    if op == "OR":
        out = ins[0].copy()
        for arr in ins[1:]:
            out |= arr
        return out
    if op == "NOR":
        return np.logical_not(eval_gate("OR", ins))
    if op == "XOR":
        out = ins[0].copy()
        for arr in ins[1:]:
            out ^= arr
        return out
    if op == "XNOR":
        return np.logical_not(eval_gate("XOR", ins))
    raise ValueError(f"unsupported gate type: {op}")


def topo_order_nodes(n_nodes: int, edges: list[tuple[int, int]]) -> tuple[list[int], list[list[int]], list[list[int]]]:
    preds = [[] for _ in range(n_nodes)]
    succs = [[] for _ in range(n_nodes)]
    indeg = [0] * n_nodes
    for u, v in edges:
        preds[v].append(u)
        succs[u].append(v)
        indeg[v] += 1
    queue = deque(i for i, deg in enumerate(indeg) if deg == 0)
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nxt in succs[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    if len(order) != n_nodes:
        seen = set(order)
        order.extend(i for i in range(n_nodes) if i not in seen)
    return order, preds, succs


def pagerank_scores(n_nodes: int, edges: list[tuple[int, int]], iterations: int = 60, damping: float = 0.85) -> np.ndarray:
    succs = [[] for _ in range(n_nodes)]
    for u, v in edges:
        succs[u].append(v)
    pr = np.full(n_nodes, 1.0 / max(1, n_nodes), dtype=float)
    base = (1.0 - damping) / max(1, n_nodes)
    for _ in range(iterations):
        nxt = np.full(n_nodes, base, dtype=float)
        dangling = 0.0
        for node, outs in enumerate(succs):
            if outs:
                share = damping * pr[node] / len(outs)
                for dst in outs:
                    nxt[dst] += share
            else:
                dangling += pr[node]
        if dangling:
            nxt += damping * dangling / max(1, n_nodes)
        pr = nxt
    return minmax(pr)


def eigen_proxy(n_nodes: int, edges: list[tuple[int, int]], iterations: int = 40) -> np.ndarray:
    neigh = [[] for _ in range(n_nodes)]
    for u, v in edges:
        neigh[u].append(v)
        neigh[v].append(u)
    vec = np.ones(n_nodes, dtype=float)
    for _ in range(iterations):
        nxt = np.zeros(n_nodes, dtype=float)
        for node, ns in enumerate(neigh):
            if ns:
                nxt[node] = float(np.sum(vec[ns]))
        norm = float(np.linalg.norm(nxt))
        if norm <= 1e-12:
            break
        vec = nxt / norm
    return minmax(vec)


def graph_feature_matrix(
    node_names: list[str],
    name_to_idx: dict[str, int],
    edges: list[tuple[int, int]],
    observed: set[str],
    dff_q: set[str],
    dff_d: set[str],
) -> tuple[np.ndarray, dict[str, dict[str, float]], int]:
    n_nodes = len(node_names)
    order, preds, succs = topo_order_nodes(n_nodes, edges)
    in_deg = np.asarray([len(preds[i]) for i in range(n_nodes)], dtype=float)
    out_deg = np.asarray([len(succs[i]) for i in range(n_nodes)], dtype=float)

    depth = np.zeros(n_nodes, dtype=float)
    for node in order:
        if preds[node]:
            depth[node] = 1.0 + max(depth[p] for p in preds[node])
    max_depth = int(float(np.max(depth))) if depth.size else 0

    obs_idx = [name_to_idx[nm] for nm in observed if nm in name_to_idx]
    dist = np.full(n_nodes, np.inf, dtype=float)
    queue = deque()
    for idx in obs_idx:
        dist[idx] = 0.0
        queue.append(idx)
    while queue:
        node = queue.popleft()
        for pred in preds[node]:
            if dist[pred] > dist[node] + 1.0:
                dist[pred] = dist[node] + 1.0
                queue.append(pred)
    finite = dist[np.isfinite(dist)]
    fallback = float(np.max(finite) + 1.0) if finite.size else 1.0
    dist[~np.isfinite(dist)] = fallback
    dist_inv = 1.0 / (1.0 + dist)

    forward_mass = np.ones(n_nodes, dtype=float)
    backward_mass = np.ones(n_nodes, dtype=float)
    for node in order:
        for succ in succs[node]:
            forward_mass[succ] += 0.5 * math.log1p(forward_mass[node])
    for node in reversed(order):
        for pred in preds[node]:
            backward_mass[pred] += 0.5 * math.log1p(backward_mass[node])
    betweenness = np.minimum(np.log1p(forward_mass), np.log1p(backward_mass))

    boundary_idx = [name_to_idx[nm] for nm in (dff_q | dff_d | observed) if nm in name_to_idx]
    near_ff = np.zeros(n_nodes, dtype=float)
    if boundary_idx:
        bdist = np.full(n_nodes, np.inf, dtype=float)
        queue = deque()
        for idx in boundary_idx:
            bdist[idx] = 0.0
            queue.append(idx)
        undirected = [sorted(set(preds[i] + succs[i])) for i in range(n_nodes)]
        while queue:
            node = queue.popleft()
            for nxt in undirected[node]:
                if bdist[nxt] > bdist[node] + 1.0:
                    bdist[nxt] = bdist[node] + 1.0
                    queue.append(nxt)
        finite_b = bdist[np.isfinite(bdist)]
        fallback_b = float(np.max(finite_b) + 1.0) if finite_b.size else 1.0
        bdist[~np.isfinite(bdist)] = fallback_b
        near_ff = 1.0 / (1.0 + bdist)

    raw_cols = {
        "in_deg": in_deg,
        "out_deg": out_deg,
        "pagerank": pagerank_scores(n_nodes, edges),
        "betweenness": betweenness,
        "eigen": eigen_proxy(n_nodes, edges),
        "dist_min_inv": dist_inv,
        "dist_avg_inv": dist_inv,
        "reconv": (in_deg > 1).astype(float),
        "near_ff": near_ff,
        "name_len": np.zeros(n_nodes, dtype=float),
        "depth": depth,
        "is_output": np.asarray([1.0 if nm in observed else 0.0 for nm in node_names], dtype=float),
    }
    x = np.zeros((n_nodes, len(FEATURE_KEYS)), dtype=np.float32)
    for col, key in enumerate(FEATURE_KEYS):
        if key in {"name_len", "is_output"}:
            vals = raw_cols[key].astype(float)
        else:
            vals = minmax(raw_cols[key])
        x[:, col] = vals.astype(np.float32)
    feature_by_name = {
        nm: {key: float(x[idx, col]) for col, key in enumerate(FEATURE_KEYS)}
        for idx, nm in enumerate(node_names)
    }
    return x, feature_by_name, max_depth


def load_bench_circuit(path: Path, suite: str) -> BenchCircuit:
    circuit = path.stem
    inputs, outputs, dff_q, dff_d, gates = parse_bench(path, suite)
    sources = set(inputs) | set(dff_q)
    topo_gates = topo_sort_gates(gates, sources)

    ordered_nodes: list[str] = []
    for group in [inputs, dff_q, [g.out for g in topo_gates], outputs, dff_d]:
        for nm in group:
            if not is_const(nm) and nm not in ordered_nodes:
                ordered_nodes.append(nm)
    for gate in topo_gates:
        for nm in gate.ins:
            if not is_const(nm) and nm not in ordered_nodes:
                ordered_nodes.append(nm)
    name_to_idx = {nm: idx for idx, nm in enumerate(ordered_nodes)}
    edges: list[tuple[int, int]] = []
    for gate in topo_gates:
        if gate.out not in name_to_idx:
            continue
        dst = name_to_idx[gate.out]
        for inp in gate.ins:
            if not is_const(inp) and inp in name_to_idx:
                edges.append((name_to_idx[inp], dst))
    observed = sorted(set(outputs) | set(dff_d))
    candidate_set = set(g.out for g in topo_gates) | set(dff_q)
    candidate_set -= set(inputs)
    candidates = [nm for nm in ordered_nodes if nm in candidate_set]
    x, feature_by_name, max_depth = graph_feature_matrix(
        ordered_nodes,
        name_to_idx,
        edges,
        set(observed),
        set(dff_q),
        set(dff_d),
    )

    prox = 0.5 * x[:, FEATURE_KEYS.index("dist_min_inv")] + 0.5 * x[:, FEATURE_KEYS.index("dist_avg_inv")]
    static_arr = minmax(prox)
    static_score = {nm: float(static_arr[name_to_idx[nm]]) for nm in candidates}
    cache_struct_score = adaptive_cache_struct_score(candidates, feature_by_name)
    stats = {
        "suite": suite,
        "circuit": circuit,
        "mode": "scan_boundary" if suite == "iscas89" else "combinational",
        "nodes": len(ordered_nodes),
        "candidate_nodes": len(candidates),
        "edges": len(edges),
        "pi": len(inputs),
        "po": len(outputs),
        "ff": len(dff_q),
        "gates": len(topo_gates),
        "depth": max_depth,
        "path": str(path),
    }
    return BenchCircuit(
        suite=suite,
        circuit=circuit,
        path=path,
        inputs=inputs,
        outputs=outputs,
        dff_q=dff_q,
        dff_d=dff_d,
        gates=gates,
        node_names=ordered_nodes,
        name_to_idx=name_to_idx,
        edges=edges,
        candidates=candidates,
        observed=[nm for nm in observed if nm in name_to_idx],
        topo_gates=topo_gates,
        x=x,
        feature_by_name={nm: feature_by_name[nm] for nm in candidates},
        static_score=static_score,
        cache_struct_score=cache_struct_score,
        stats=stats,
    )


def golden_values(circuit: BenchCircuit, vectors: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    values = [np.zeros(vectors, dtype=bool) for _ in circuit.node_names]
    for nm in list(circuit.inputs) + list(circuit.dff_q):
        if nm in circuit.name_to_idx:
            values[circuit.name_to_idx[nm]] = rng.integers(0, 2, size=vectors, dtype=np.int8).astype(bool)
    for gate in circuit.topo_gates:
        ins = [
            const_value(inp, vectors) if is_const(inp) else values[circuit.name_to_idx[inp]]
            for inp in gate.ins
            if is_const(inp) or inp in circuit.name_to_idx
        ]
        values[circuit.name_to_idx[gate.out]] = eval_gate(gate.op, ins)
    return values


def simulate_fault_counts(circuit: BenchCircuit, vectors: int, seed: int) -> dict[str, int]:
    values = golden_values(circuit, vectors, seed)
    observed_idx = [circuit.name_to_idx[nm] for nm in circuit.observed if nm in circuit.name_to_idx]
    gate_by_out = {circuit.name_to_idx[g.out]: i for i, g in enumerate(circuit.topo_gates) if g.out in circuit.name_to_idx}
    gate_succs: dict[int, list[int]] = defaultdict(list)
    for gid, gate in enumerate(circuit.topo_gates):
        for inp in gate.ins:
            if not is_const(inp) and inp in circuit.name_to_idx:
                gate_succs[circuit.name_to_idx[inp]].append(gid)
    del gate_by_out

    counts: dict[str, int] = {}
    for name in circuit.candidates:
        idx = circuit.name_to_idx[name]
        node_count = 0
        for stuck in (False, True):
            stuck_arr = np.full(vectors, stuck, dtype=bool)
            if np.array_equal(values[idx], stuck_arr):
                continue
            affected: dict[int, np.ndarray] = {idx: stuck_arr}
            diff_obs = np.zeros(vectors, dtype=bool)
            if idx in observed_idx:
                diff_obs |= values[idx] != stuck_arr
            queue = deque(gate_succs.get(idx, []))
            queued = set(queue)
            while queue:
                gid = queue.popleft()
                queued.discard(gid)
                gate = circuit.topo_gates[gid]
                out_idx = circuit.name_to_idx[gate.out]
                ins = []
                for inp in gate.ins:
                    if is_const(inp):
                        ins.append(const_value(inp, vectors))
                    else:
                        inp_idx = circuit.name_to_idx[inp]
                        ins.append(affected.get(inp_idx, values[inp_idx]))
                new_val = eval_gate(gate.op, ins)
                if not np.array_equal(new_val, values[out_idx]):
                    old_val = affected.get(out_idx)
                    if old_val is None or not np.array_equal(old_val, new_val):
                        affected[out_idx] = new_val
                        if out_idx in observed_idx:
                            diff_obs |= values[out_idx] != new_val
                        for succ_gid in gate_succs.get(out_idx, []):
                            if succ_gid not in queued:
                                queue.append(succ_gid)
                                queued.add(succ_gid)
            node_count += int(np.count_nonzero(diff_obs))
        if node_count:
            counts[name] = node_count
    return counts


def random_ratios(counts: dict[str, int], names: list[str], budgets: list[float], samples: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n_nodes = len(names)
    count_arr = np.asarray([float(counts.get(n, 0) or 0) for n in names], dtype=float)
    ks = [max(1, min(n_nodes, math.ceil(float(budget) * n_nodes))) for budget in budgets]
    kmax = max(ks)
    oracles = [oracle_fault_instances(counts, k) for k in ks]
    out: list[float] = []
    for _ in range(samples):
        sampled = rng.choice(n_nodes, size=kmax, replace=False)
        prefix_hits = np.cumsum(count_arr[sampled])
        vals = []
        for k, oracle in zip(ks, oracles):
            hit = float(prefix_hits[k - 1])
            vals.append(hit / oracle if oracle else 1.0)
        out.append(float(mean(vals)))
    return out


def row_ratio(row: dict[str, Any]) -> float:
    oracle = float(row.get("fault_instance_oracle", 0.0) or 0.0)
    selected = float(row.get("fault_instance_selected", 0.0) or 0.0)
    return selected / oracle if oracle else 1.0


def run_one(args: argparse.Namespace, suite: str, bench_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    circuit = load_bench_circuit(bench_path, suite)
    t0 = time.perf_counter()

    t_gnn = time.perf_counter()
    gnn = score_gnn(
        circuit.candidates,
        circuit.name_to_idx,
        circuit.x,
        circuit.edges,
        circuit.cache_struct_score,
        epochs=args.epochs,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        device=args.device,
        train_node_cap=args.train_node_cap,
        seed=args.seed,
    )
    gnn_seconds = time.perf_counter() - t_gnn

    t_rank = time.perf_counter()
    rank = build_global_rank(
        circuit.candidates,
        circuit.static_score,
        circuit.cache_struct_score,
        gnn.rank,
        gnn.diagnostics,
        feature_by_name=circuit.feature_by_name,
        name_to_idx=circuit.name_to_idx,
        edges=circuit.edges,
        x_np=circuit.x,
    )
    rank_seconds = time.perf_counter() - t_rank

    t_fi = time.perf_counter()
    fi_seed = int(args.vector_seed) + sum(ord(ch) for ch in circuit.circuit)
    counts = simulate_fault_counts(circuit, int(args.vectors), fi_seed)
    fi_seconds = time.perf_counter() - t_fi

    static_order = sorted(circuit.candidates, key=lambda n: circuit.static_score.get(n, 0.0), reverse=True)
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        k = max(1, min(len(circuit.candidates), math.ceil(float(budget) * len(circuit.candidates))))
        oracle = oracle_fault_instances(counts, k)
        common = {
            "suite": suite,
            "circuit": circuit.circuit,
            "label_mode": "external_random_vector_stuck_at",
            "scan_mode": circuit.stats["mode"],
            "budget": budget,
            "pool_mode": "global_prefix",
            "n_nodes": len(circuit.candidates),
            "budget_nodes": k,
            "vectors": int(args.vectors),
            "global_rank_prefix_consistent": True,
            "chosen_family": rank.diagnostics.get("chosen_family", ""),
            "family_reason": rank.diagnostics.get("family_reason", ""),
            "family_selector_mode": rank.diagnostics.get("family_selector_mode", ""),
            "family_peer_frontier_overlap": rank.diagnostics.get("family_peer_frontier_overlap", ""),
            "family_gnn_frontier_overlap": rank.diagnostics.get("family_gnn_frontier_overlap", ""),
            "family_cache_frontier_overlap": rank.diagnostics.get("family_cache_frontier_overlap", ""),
            "family_static_core_overlap": rank.diagnostics.get("family_static_core_overlap", ""),
            "gnn_nonrandom": rank.diagnostics.get("gnn_nonrandom", ""),
            "gnn_struct_agree": rank.diagnostics.get("gnn_struct_agree", ""),
            "gnn_reliable": rank.diagnostics.get("gnn_reliable", ""),
            "gnn_participated": float(rank.diagnostics.get("effective_gnn_weight", 0.0) or 0.0) > 0.0,
            "distance_guard_active": rank.diagnostics.get("distance_guard_active", ""),
            "distance_guard_mode": rank.diagnostics.get("distance_guard_mode", ""),
        }
        static_sel = topk(static_order, k)
        method_sel = topk(rank.ranked_nodes, k)
        rows.append({
            **common,
            "method": "pure_static_proximity",
            "selected": len(static_sel),
            **eval_selection(static_sel, counts, oracle),
        })
        rows.append({
            **common,
            "method": "segr_structure_derived_selector",
            "selected": len(method_sel),
            **eval_selection(method_sel, counts, oracle),
        })

    perf = {
        **circuit.stats,
        "fi_seconds": fi_seconds,
        "gnn_seconds": gnn_seconds,
        "rank_seconds": rank_seconds,
        "total_seconds": time.perf_counter() - t0,
        **rank.diagnostics,
    }
    for row in rank.node_debug:
        row["suite"] = suite
        row["circuit"] = circuit.circuit
    return rows, perf, rank.node_debug, circuit.stats, counts


def summarize(results: list[dict[str, Any]], perf_rows: list[dict[str, Any]], random_rows: list[dict[str, Any]], out_dir: Path) -> None:
    groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in results:
        key = (str(row["suite"]), str(row["circuit"]), str(row["budget"]))
        if row["method"] == "pure_static_proximity":
            groups[key]["static"] = row
        else:
            groups[key]["method"] = row

    row_out: list[dict[str, Any]] = []
    by_circuit: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    eps = 1e-9
    for (suite, circuit, budget), pair in sorted(groups.items()):
        static = pair.get("static")
        method = pair.get("method")
        if not static or not method:
            continue
        static_value = float(static["fault_instance_selected"])
        method_value = float(method["fault_instance_selected"])
        oracle_value = float(method["fault_instance_oracle"])
        loss = method_value < static_value - eps
        ratio = method_value / oracle_value if oracle_value > eps else (1.0 if not loss else -1.0)
        if oracle_value <= static_value + eps:
            closure = 1.0 if not loss else -1.0
        else:
            closure = (method_value - static_value) / (oracle_value - static_value)
        item = {
            "suite": suite,
            "circuit": circuit,
            "budget": budget,
            "static_value": static_value,
            "method_value": method_value,
            "oracle_value": oracle_value,
            "row_ideal_ratio_raw": ratio,
            "row_closure_raw": closure,
            "loss": loss,
            "chosen_family": method.get("chosen_family", ""),
            "family_reason": method.get("family_reason", ""),
        }
        row_out.append(item)
        by_circuit[(suite, circuit)].append(item)

    random_by_circuit = {(str(r["suite"]), str(r["circuit"])): r for r in random_rows}
    circuits: list[dict[str, Any]] = []
    for (suite, circuit), items in sorted(by_circuit.items()):
        ratio = mean(float(x["row_ideal_ratio_raw"]) for x in items)
        closure = mean(float(x["row_closure_raw"]) for x in items)
        losses = sum(1 for x in items if bool(x["loss"]))
        rand = random_by_circuit.get((suite, circuit), {})
        gate = float(rand.get("gate", 0.5) or 0.5)
        circuits.append({
            "suite": suite,
            "circuit": circuit,
            "circuit_ideal_ratio_raw_mean": ratio,
            "circuit_closure_raw_mean": closure,
            "loss_rows": losses,
            "random_gate": gate,
            "random_passed": bool(rand.get("passed", False)),
            "chosen_family": items[0]["chosen_family"],
            "family_reason": items[0]["family_reason"],
        })

    suite_rows: list[dict[str, Any]] = []
    for suite in sorted(set(row["suite"] for row in circuits)):
        suite_items = [row for row in circuits if row["suite"] == suite]
        suite_rows.append({
            "suite": suite,
            "circuits": len(suite_items),
            "macro_ideal_ratio_raw": mean(float(x["circuit_ideal_ratio_raw_mean"]) for x in suite_items),
            "macro_closure_raw": mean(float(x["circuit_closure_raw_mean"]) for x in suite_items),
            "loss_rows": sum(int(x["loss_rows"]) for x in suite_items),
            "random_gate_fail": sum(1 for x in suite_items if not bool(x["random_passed"])),
            "runtime_seconds": sum(float(p.get("total_seconds", 0.0) or 0.0) for p in perf_rows if p.get("suite") == suite),
        })

    summary = {
        "combined_circuits": len(circuits),
        "combined_macro_ideal_ratio_raw": mean(float(x["circuit_ideal_ratio_raw_mean"]) for x in circuits) if circuits else 0.0,
        "combined_macro_closure_raw": mean(float(x["circuit_closure_raw_mean"]) for x in circuits) if circuits else 0.0,
        "combined_loss_rows": sum(int(x["loss_rows"]) for x in circuits),
        "combined_random_gate_fail": sum(1 for x in circuits if not bool(x["random_passed"])),
        "runtime_total_seconds": sum(float(p.get("total_seconds", 0.0) or 0.0) for p in perf_rows),
        "suites": suite_rows,
        "note": "ISCAS89 uses scan-boundary mode with DFF Q as pseudo-PI and DFF D as pseudo-PO.",
    }

    write_rows(out_dir / "rows.csv", row_out)
    write_rows(out_dir / "circuits.csv", circuits)
    write_rows(out_dir / "suites.csv", suite_rows)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-root", type=Path, default=Path("data/external_benchmarks"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--perf", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--vectors", type=int, default=128)
    parser.add_argument("--vector-seed", type=int, default=5089)
    parser.add_argument("--random-samples", type=int, default=1000)
    parser.add_argument("--random-seed", type=int, default=85289)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--train-node-cap", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--only", nargs="*", default=None, help="Optional smoke-test filter; omit for formal runs.")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test prefix limit; omit for formal runs.")
    args = parser.parse_args()

    circuits: list[tuple[str, Path]] = []
    for name in ISCAS85:
        circuits.append(("iscas85", args.bench_root / "iscas85" / "bench" / f"{name}.bench"))
    for name in ISCAS89:
        circuits.append(("iscas89", args.bench_root / "iscas89" / "bench" / f"{name}.bench"))
    if args.only:
        wanted = set(args.only)
        circuits = [
            (suite, path)
            for suite, path in circuits
            if path.stem in wanted or safe_id(path.stem) in wanted or f"{suite}/{path.stem}" in wanted
        ]
    if args.limit > 0:
        circuits = circuits[: args.limit]

    all_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    random_rows: list[dict[str, Any]] = []
    args.debug_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_dir.mkdir(parents=True, exist_ok=True)

    for suite, path in circuits:
        if not path.exists():
            stats_rows.append({"suite": suite, "circuit": path.stem, "status": "missing", "path": str(path)})
            continue
        try:
            rows, perf, node_debug, stats, counts = run_one(args, suite, path)
            all_rows.extend(rows)
            perf_rows.append(perf)
            stats_rows.append({**stats, "status": "ok"})
            write_rows(args.debug_dir / f"{safe_id(suite + '_' + path.stem)}_node_debug.csv", node_debug)
            method_vals = [row_ratio(r) for r in rows if r["method"] != "pure_static_proximity"]
            rand_vals = random_ratios(counts, [r for r in load_bench_circuit(path, suite).candidates], BUDGETS, int(args.random_samples), int(args.random_seed))
            random_mean = float(np.mean(rand_vals))
            random_p05 = float(np.quantile(rand_vals, 0.05))
            random_p95 = float(np.quantile(rand_vals, 0.95))
            gate = max(0.50, random_mean)
            method_ratio = float(mean(method_vals)) if method_vals else 0.0
            random_rows.append({
                "suite": suite,
                "circuit": path.stem,
                "method_ratio": method_ratio,
                "random_mean": random_mean,
                "random_p05": random_p05,
                "random_p95": random_p95,
                "floor": 0.50,
                "gate": gate,
                "passed": method_ratio + 1e-12 >= gate,
            })
            print(f"{suite}/{path.stem}: ratio={method_ratio:.6f} random_gate={gate:.6f} family={perf.get('chosen_family', '')}")
        except Exception as exc:
            stats_rows.append({"suite": suite, "circuit": path.stem, "status": "failed", "error": repr(exc), "path": str(path)})
            print(f"{suite}/{path.stem}: FAILED {exc!r}")

    write_rows(args.output, all_rows)
    write_rows(args.perf, perf_rows)
    write_rows(args.analysis_dir / "benchmark_stats.csv", stats_rows)
    write_rows(args.analysis_dir / "random_gate_sampling.csv", random_rows)
    summarize(all_rows, perf_rows, random_rows, args.analysis_dir)
    (args.debug_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Wrote {len(all_rows)} rows to {args.output}")
    print((args.analysis_dir / "summary.json").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
