from __future__ import annotations

from collections import deque

import numpy as np

from .data_io import FEATURE_KEYS, minmax, rank01


def _norm_map(names: list[str], arr: np.ndarray) -> dict[str, float]:
    vals = minmax(np.asarray(arr, dtype=float))
    return {name: float(vals[i]) for i, name in enumerate(names)}


def _topo_order(n_nodes: int, edges: list[tuple[int, int]]) -> tuple[list[int], list[list[int]], list[list[int]]]:
    preds = [[] for _ in range(n_nodes)]
    succs = [[] for _ in range(n_nodes)]
    indeg = [0] * n_nodes
    for src, dst in edges:
        if 0 <= src < n_nodes and 0 <= dst < n_nodes:
            succs[src].append(dst)
            preds[dst].append(src)
            indeg[dst] += 1
    queue = deque([idx for idx, degree in enumerate(indeg) if degree == 0])
    order: list[int] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for dst in succs[node]:
            indeg[dst] -= 1
            if indeg[dst] == 0:
                queue.append(dst)
    if len(order) != n_nodes:
        seen = set(order)
        order.extend(idx for idx in range(n_nodes) if idx not in seen)
    return order, preds, succs


def _wl_scores(
    names: list[str],
    name_to_idx: dict[str, int],
    edges: list[tuple[int, int]],
    feature_by_name: dict[str, dict[str, float]],
    *,
    iterations: int = 3,
) -> dict[str, dict[str, float]]:
    n_nodes = max(name_to_idx.values(), default=-1) + 1
    order, preds, succs = _topo_order(n_nodes, edges)
    del order
    in_deg = np.asarray([len(preds[i]) for i in range(n_nodes)], dtype=np.int32)
    out_deg = np.asarray([len(succs[i]) for i in range(n_nodes)], dtype=np.int32)
    labels = np.asarray([min(int(in_deg[i]), 8) * 16 + min(int(out_deg[i]), 15) for i in range(n_nodes)], dtype=np.int64)
    valid_idx = np.asarray([name_to_idx[nm] for nm in names], dtype=np.int64)
    idx_frac = valid_idx.astype(float) / max(1.0, n_nodes - 1.0)
    idx_edge = np.abs(idx_frac - 0.5) * 2.0
    out: dict[str, dict[str, float]] = {}
    for step in range(1, iterations + 1):
        label_map: dict[tuple[int, tuple[int, ...], tuple[int, ...]], int] = {}
        next_labels = np.zeros(n_nodes, dtype=np.int64)
        for node in range(n_nodes):
            key = (
                int(labels[node]),
                tuple(sorted(int(labels[p]) for p in preds[node])),
                tuple(sorted(int(labels[s]) for s in succs[node])),
            )
            label_id = label_map.get(key)
            if label_id is None:
                label_id = len(label_map) + 1
                label_map[key] = label_id
            next_labels[node] = label_id
        labels = next_labels
        valid_labels = labels[valid_idx]
        unique, inverse, counts = np.unique(valid_labels, return_inverse=True, return_counts=True)
        class_size = counts[inverse].astype(float)
        out[f"wl{step}_common"] = _norm_map(names, class_size)
        for feature in ["depth", "pagerank", "dist_avg_inv", "out_deg"]:
            arr = np.asarray([feature_by_name[nm].get(feature, 0.0) for nm in names], dtype=float)
            sums = np.zeros(len(unique), dtype=float)
            np.add.at(sums, inverse, arr)
            out[f"wl{step}_role_{feature}"] = _norm_map(names, sums[inverse] / np.maximum(1.0, class_size))
        sums = np.zeros(len(unique), dtype=float)
        np.add.at(sums, inverse, idx_edge)
        out[f"wl{step}_role_index_edge"] = _norm_map(names, sums[inverse] / np.maximum(1.0, class_size))
    return out


def _graph_scores(
    names: list[str],
    name_to_idx: dict[str, int],
    edges: list[tuple[int, int]],
) -> dict[str, dict[str, float]]:
    n_nodes = max(name_to_idx.values(), default=-1) + 1
    order, preds, succs = _topo_order(n_nodes, edges)
    valid_idx = np.asarray([name_to_idx[nm] for nm in names], dtype=np.int64)
    topo_pos = np.zeros(n_nodes, dtype=float)
    for rank, idx in enumerate(order):
        topo_pos[idx] = rank / max(1.0, n_nodes - 1.0)
    forward_mass = np.ones(n_nodes, dtype=float)
    backward_mass = np.ones(n_nodes, dtype=float)
    sink_dist = np.zeros(n_nodes, dtype=float)
    for node in order:
        if preds[node]:
            forward_mass[node] += 0.5 * float(np.sum(np.log1p(forward_mass[preds[node]])))
    for node in reversed(order):
        if succs[node]:
            backward_mass[node] += 0.5 * float(np.sum(np.log1p(backward_mass[succs[node]])))
            sink_dist[node] = 1.0 + min(sink_dist[s] for s in succs[node])
    return {
        "topo_index": _norm_map(names, topo_pos[valid_idx]),
        "topo_edge": _norm_map(names, np.abs(topo_pos[valid_idx] - 0.5) * 2.0),
        "forward_mass": _norm_map(names, np.log1p(forward_mass[valid_idx])),
        "backward_mass": _norm_map(names, np.log1p(backward_mass[valid_idx])),
        "mass_balance": _norm_map(names, np.minimum(np.log1p(forward_mass[valid_idx]), np.log1p(backward_mass[valid_idx]))),
        "sink_near": _norm_map(names, 1.0 / (1.0 + sink_dist[valid_idx])),
    }


def build_structural_families(
    names: list[str],
    name_to_idx: dict[str, int],
    edges: list[tuple[int, int]],
    feature_by_name: dict[str, dict[str, float]],
    cache_struct_score: dict[str, float],
    gnn_rank: dict[str, float],
    final_score: dict[str, float],
) -> dict[str, dict[str, float]]:
    families: dict[str, dict[str, float]] = {
        "cache_struct": cache_struct_score,
        "gnn_rank": gnn_rank,
        "final_score": final_score,
        "final_rank": rank01(final_score, names),
    }
    for key in FEATURE_KEYS:
        if key == "name_len":
            continue
        raw = np.asarray([float(feature_by_name.get(n, {}).get(key, 0.0) or 0.0) for n in names], dtype=float)
        families[key] = _norm_map(names, raw)
    families["inv_depth"] = {n: 1.0 - families["depth"].get(n, 0.0) for n in names}

    n = max(1, len(names))
    idx = np.arange(n, dtype=float) / max(1.0, n - 1.0)
    full_max_idx = max(1.0, float(max((name_to_idx[nm] for nm in names), default=1)))
    orig_idx = np.asarray([float(name_to_idx[nm]) / full_max_idx for nm in names], dtype=float)
    families["node_index"] = _norm_map(names, idx)
    families["inv_node_index"] = _norm_map(names, 1.0 - idx)
    families["index_mid"] = _norm_map(names, 1.0 - np.abs(idx - 0.5) * 2.0)
    families["orig_index_mid"] = _norm_map(names, 1.0 - np.abs(orig_idx - 0.5) * 2.0)
    families["orig_index_edge"] = _norm_map(names, np.abs(orig_idx - 0.5) * 2.0)

    families.update(_graph_scores(names, name_to_idx, edges))
    families.update(_wl_scores(names, name_to_idx, edges, feature_by_name))

    def product(out: str, left: str, right: str) -> None:
        if left in families and right in families:
            families[out] = {n: families[left].get(n, 0.0) * families[right].get(n, 0.0) for n in names}

    product("dist_pr", "dist_avg_inv", "pagerank")
    product("dist_eig", "dist_avg_inv", "eigen")
    product("idx_mid_cache", "index_mid", "cache_struct")
    product("orig_idx_mid_cache", "orig_index_mid", "cache_struct")
    product("orig_idx_inv_dist", "inv_node_index", "dist_avg_inv")
    product("idx_inv_dist", "inv_node_index", "dist_avg_inv")
    product("orig_idx_edge_depth_inv", "orig_index_edge", "inv_depth")
    product("idx_edge_depth_inv", "orig_index_edge", "inv_depth")
    product("topo_edge_pr", "topo_edge", "pagerank")
    product("wl1_role_depth_pr", "wl1_role_depth", "pagerank")
    product("wl1_role_index_edge_dist", "wl1_role_index_edge", "dist_avg_inv")
    product("wl3_common_dist", "wl3_common", "dist_avg_inv")
    product("wl3_role_depth_gnn", "wl3_role_depth", "gnn_rank")
    product("forward_mass_pr", "forward_mass", "pagerank")
    product("sink_reach_near_pr", "sink_near", "pagerank")
    product("pdom_dist", "sink_near", "dist_avg_inv")
    families["union_dist_pr_topo_edge_pr"] = {
        n: max(families["dist_pr"].get(n, 0.0), families["topo_edge_pr"].get(n, 0.0))
        for n in names
    }
    families["union_orig_idx_edge_pr_topo_edge_pr"] = {
        n: max(
            families["orig_index_edge"].get(n, 0.0) * families["pagerank"].get(n, 0.0),
            families["topo_edge_pr"].get(n, 0.0),
        )
        for n in names
    }
    for term in ["orig_idx_edge_depth_inv", "idx_edge_depth_inv", "orig_idx_inv_dist", "idx_inv_dist"]:
        product(f"role_depth_resonance_{term}", "wl1_role_depth_pr", term)
    families["role_depth_biindex_dist"] = {
        n: families["wl1_role_depth_pr"].get(n, 0.0)
        * max(families["orig_idx_inv_dist"].get(n, 0.0), families["dist_pr"].get(n, 0.0))
        for n in names
    }
    families["role_depth_frontier_idx_edge_depth_inv"] = {
        n: min(families["wl1_role_depth_pr"].get(n, 0.0), families["idx_edge_depth_inv"].get(n, 0.0))
        for n in names
    }
    return families
