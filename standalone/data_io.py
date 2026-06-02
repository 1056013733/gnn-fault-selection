from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


FEATURE_KEYS = [
    "in_deg",
    "out_deg",
    "pagerank",
    "betweenness",
    "eigen",
    "dist_min_inv",
    "dist_avg_inv",
    "reconv",
    "near_ff",
    "name_len",
    "depth",
    "is_output",
]
CACHE_STRUCT_EXCLUDED_FEATURES = {"name_len"}


EPFL20 = [
    "adder",
    "arbiter",
    "bar",
    "cavlc",
    "ctrl",
    "dec",
    "div",
    "hyp",
    "i2c",
    "int2float",
    "log2",
    "max",
    "mem_ctrl",
    "multiplier",
    "priority",
    "router",
    "sin",
    "sqrt",
    "square",
    "voter",
]


@dataclass
class CircuitData:
    circuit: str
    node_names: list[str]
    name_to_idx: dict[str, int]
    edges: list[tuple[int, int]]
    x: np.ndarray
    feature_by_name: dict[str, dict[str, float]]
    node_type_by_idx: dict[int, str]
    node_type_by_name: dict[str, str]
    static_score: dict[str, float]
    cache_struct_score: dict[str, float]
    fi: dict[str, Any]


def minmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.astype(float)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def rank01(values: dict[str, float], names: list[str]) -> dict[str, float]:
    ranked = sorted(names, key=lambda n: float(values.get(n, 0.0) or 0.0))
    denom = max(1, len(ranked) - 1)
    out: dict[str, float] = {}
    i = 0
    while i < len(ranked):
        value = float(values.get(ranked[i], 0.0) or 0.0)
        j = i + 1
        while j < len(ranked) and float(values.get(ranked[j], 0.0) or 0.0) == value:
            j += 1
        rank_value = ((i + j - 1) * 0.5) / denom
        for name in ranked[i:j]:
            out[name] = rank_value
        i = j
    return out


def _positive_alignment(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    lpos = np.maximum(left - float(np.mean(left)), 0.0)
    rpos = np.maximum(right - float(np.mean(right)), 0.0)
    denom = float(np.linalg.norm(lpos) * np.linalg.norm(rpos))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(lpos, rpos) / denom)


def _unique_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return len({round(float(v), 12) for v in values}) / float(values.size)


def adaptive_cache_struct_score(
    names: list[str],
    feature_by_name: dict[str, dict[str, float]],
) -> dict[str, float]:
    rank_columns: list[np.ndarray] = []
    for key in FEATURE_KEYS:
        if key in CACHE_STRUCT_EXCLUDED_FEATURES:
            continue
        values = {name: float(feature_by_name.get(name, {}).get(key, 0.0) or 0.0) for name in names}
        arr = np.asarray([values[name] for name in names], dtype=float)
        if arr.size == 0 or float(np.nanmax(arr)) <= float(np.nanmin(arr)):
            continue
        ranks = rank01(values, names)
        rank_columns.append(np.asarray([ranks[name] for name in names], dtype=float))

    if not rank_columns:
        return {name: 0.0 for name in names}

    matrix = np.vstack(rank_columns).T
    consensus = np.mean(matrix, axis=1)
    feature_weights = np.asarray(
        [
            _positive_alignment(matrix[:, idx], consensus) + _unique_fraction(matrix[:, idx])
            for idx in range(matrix.shape[1])
        ],
        dtype=float,
    )
    total = float(np.sum(feature_weights))
    if total <= 0.0:
        score = consensus
    else:
        score = matrix @ (feature_weights / total)
    normalized = minmax(score)
    return {name: float(normalized[idx]) for idx, name in enumerate(names)}


def find_feature_cache(root: Path, circuit: str) -> Path:
    matches = sorted((root / "feature_cache").glob(f"{circuit}_v1_*.pkl"))
    matches = [p for p in matches if ".pre_restore_" not in p.name]
    if not matches:
        raise FileNotFoundError(f"missing feature cache for {circuit}")
    return matches[0]


def load_feature_cache(path: Path) -> tuple[dict[int, dict[str, float]], dict[str, Any]]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "feats" not in obj or "nk_graph" not in obj:
        raise ValueError(f"unsupported feature cache: {path}")
    return obj["feats"], obj["nk_graph"]


def build_matrix(feats: dict[int, dict[str, float]], n_nodes: int) -> np.ndarray:
    x = np.zeros((n_nodes, len(FEATURE_KEYS)), dtype=np.float32)
    for idx in range(n_nodes):
        item = feats.get(idx, {})
        x[idx, :] = [float(item.get(key, 0.0) or 0.0) for key in FEATURE_KEYS]
    return x


def structural_candidate_names(
    node_names: list[str],
    node_type_by_idx: dict[int, str],
    edges: list[tuple[int, int]],
) -> list[str]:
    has_pred = {int(dst) for _src, dst in edges}
    out: list[str] = []
    for idx, name in enumerate(node_names):
        if idx not in has_pred:
            continue
        if node_type_by_idx.get(idx, "") != "wire":
            continue
        if str(name).startswith("\\"):
            continue
        out.append(str(name))
    return out


def load_circuit(root: Path, fi_root: str, circuit: str, *, load_fi: bool = True) -> CircuitData:
    feats, graph = load_feature_cache(find_feature_cache(root, circuit))
    n_nodes = int(graph.get("n_nodes", len(feats)))
    raw_names = graph.get("node_names", {})
    raw_types = graph.get("node_types", {})
    node_names = [str(raw_names.get(i, i)) for i in range(n_nodes)]
    name_to_idx = {name: idx for idx, name in enumerate(node_names)}
    edges = [(int(u), int(v)) for u, v in graph.get("edges", [])]
    x = build_matrix(feats, n_nodes)
    node_type_by_idx = {idx: str(raw_types.get(idx, "")) for idx in range(n_nodes)}
    if load_fi:
        fi_path = root / fi_root / circuit / "full_injection_results.json"
        fi = json.loads(fi_path.read_text(encoding="utf-8", errors="ignore"))
        valid_names = [n for n in node_names if n in fi]
    else:
        fi = {}
        valid_names = structural_candidate_names(node_names, node_type_by_idx, edges)
    feature_by_name = {n: {k: float(feats[name_to_idx[n]].get(k, 0.0) or 0.0) for k in FEATURE_KEYS} for n in valid_names}
    node_type_by_name = {n: node_type_by_idx.get(name_to_idx[n], "") for n in valid_names}

    prox = 0.5 * x[:, 5] + 0.5 * x[:, 6]
    static_arr = minmax(prox)
    static_score = {n: float(static_arr[name_to_idx[n]]) for n in valid_names}
    cache_struct_score = adaptive_cache_struct_score(valid_names, feature_by_name)
    return CircuitData(
        circuit=circuit,
        node_names=valid_names,
        name_to_idx=name_to_idx,
        edges=edges,
        x=x,
        feature_by_name=feature_by_name,
        node_type_by_idx=node_type_by_idx,
        node_type_by_name=node_type_by_name,
        static_score=static_score,
        cache_struct_score=cache_struct_score,
        fi=fi,
    )


def edge_index_tensor(edges: list[tuple[int, int]]) -> torch.Tensor:
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()
