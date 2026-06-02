from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data_io import FEATURE_KEYS, rank01
from .structural_features import build_structural_families


CANDIDATE_FAMILIES = [
    "static_proximity",
    "dist_avg_inv",
    "pagerank",
    "betweenness",
    "eigen",
    "out_deg",
    "inv_depth",
    "cache_struct",
    "gnn_rank",
    "final_score",
    "final_rank",
    "index_mid",
    "idx_mid_cache",
    "dist_pr",
    "union_orig_idx_edge_pr_topo_edge_pr",
    "union_dist_pr_topo_edge_pr",
    "wl1_role_index_edge_dist",
    "wl3_common_dist",
    "wl3_role_depth_gnn",
    "forward_mass_pr",
    "sink_reach_near_pr",
    "pdom_dist",
    "mass_balance",
    "role_depth_resonance_orig_idx_edge_depth_inv",
    "role_depth_resonance_idx_edge_depth_inv",
    "role_depth_resonance_orig_idx_inv_dist",
    "role_depth_resonance_idx_inv_dist",
    "role_depth_frontier_idx_edge_depth_inv",
    "role_depth_biindex_dist",
]
GNN_FAMILY = CANDIDATE_FAMILIES[8]


@dataclass
class RankResult:
    ranked_nodes: list[str]
    node_debug: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def stable_order(names: list[str], scores: dict[str, float], *, preserve_ties: bool = False) -> list[str]:
    if preserve_ties:
        return sorted(names, key=lambda n: float(scores.get(n, 0.0) or 0.0), reverse=True)
    return sorted(names, key=lambda n: (float(scores.get(n, 0.0) or 0.0), n), reverse=True)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / max(1, len(a | b))


def ensemble_top_set(ranking: list[str], fraction: float) -> set[str]:
    import math

    return set(ranking[: max(1, math.ceil(float(fraction) * len(ranking)))])


def rank_values(names: list[str], scores: dict[str, float]) -> np.ndarray:
    return np.asarray([float(scores.get(name, 0.0) or 0.0) for name in names], dtype=float)


def positive_alignment(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    lpos = np.maximum(left - float(np.mean(left)), 0.0)
    rpos = np.maximum(right - float(np.mean(right)), 0.0)
    denom = float(np.linalg.norm(lpos) * np.linalg.norm(rpos))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(lpos, rpos) / denom)


def unique_fraction(names: list[str], scores: dict[str, float]) -> float:
    if not names:
        return 0.0
    vals = {float(scores.get(name, 0.0) or 0.0) for name in names}
    return len(vals) / float(len(names))


def adaptive_signal_weights(
    names: list[str],
    static_rank: dict[str, float],
    cache_rank: dict[str, float],
    gnn_rank: dict[str, float],
) -> tuple[float, float]:
    static_vals = rank_values(names, static_rank)
    cache_vals = rank_values(names, cache_rank)
    gnn_vals = rank_values(names, gnn_rank)
    gnn_signal = (
        positive_alignment(gnn_vals, static_vals)
        + positive_alignment(gnn_vals, cache_vals)
        + unique_fraction(names, gnn_rank)
    )
    cache_signal = (
        positive_alignment(cache_vals, static_vals)
        + positive_alignment(cache_vals, gnn_vals)
        + unique_fraction(names, cache_rank)
    )
    total = gnn_signal + cache_signal
    if total <= 0.0:
        return 1.0, 1.0
    return gnn_signal, cache_signal


def adaptive_residual_step(values: dict[str, float], names: list[str]) -> float:
    uniq = sorted({float(values.get(name, 0.0) or 0.0) for name in names})
    gaps = np.diff(np.asarray(uniq, dtype=float))
    positive = gaps[gaps > 0.0]
    if positive.size == 0:
        return 0.0
    return float(np.mean(positive))


def structure_effective_dimension(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 1.0
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    denom = float(np.dot(singular, singular))
    if denom <= 0.0:
        return 1.0
    total = float(np.sum(singular))
    return max(1.0, (total * total) / denom)


def structural_rank_matrix(names: list[str], feature_by_name: dict[str, dict[str, float]]) -> np.ndarray:
    columns: list[np.ndarray] = []
    schema_excluded = {FEATURE_KEYS[-3], FEATURE_KEYS[-1]}
    for key in FEATURE_KEYS:
        if key in schema_excluded:
            continue
        values = {name: float(feature_by_name.get(name, {}).get(key, 0.0) or 0.0) for name in names}
        arr = np.asarray([values[name] for name in names], dtype=float)
        if arr.size == 0 or float(np.nanmax(arr)) <= float(np.nanmin(arr)):
            continue
        ranks = rank01(values, names)
        columns.append(np.asarray([ranks[name] for name in names], dtype=float))
    if not columns:
        return np.ones((len(names), 1), dtype=float)
    return np.vstack(columns).T


def structural_frontier_profile(names: list[str], feature_by_name: dict[str, dict[str, float]]) -> dict[str, float]:
    matrix = structural_rank_matrix(names, feature_by_name)
    active_features = max(1.0, float(matrix.shape[1]))
    eff_dim = structure_effective_dimension(matrix)
    frontier_fraction = 1.0 / eff_dim
    core_fraction = 1.0 / (eff_dim + active_features)
    tail_quantile = 1.0 - (1.0 / (eff_dim * active_features))
    return {
        "structural_effective_dim": eff_dim,
        "structural_active_features": active_features,
        "frontier_fraction": clamp(frontier_fraction),
        "core_fraction": clamp(core_fraction),
        "tail_quantile": clamp(tail_quantile),
    }


def family_metrics(
    rankings: dict[str, list[str]],
    family_scores: dict[str, dict[str, float]],
    names: list[str],
    feature_by_name: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    frontier_profile = structural_frontier_profile(names, feature_by_name)
    frontier_fraction = frontier_profile["frontier_fraction"]
    core_fraction = frontier_profile["core_fraction"]
    tail_quantile = frontier_profile["tail_quantile"]
    frontier = {feature: ensemble_top_set(ranking, frontier_fraction) for feature, ranking in rankings.items()}
    core = {feature: ensemble_top_set(ranking, core_fraction) for feature, ranking in rankings.items()}
    metrics: dict[str, dict[str, float]] = {}
    for feature in CANDIDATE_FAMILIES:
        vals = np.asarray([float(family_scores[feature].get(n, 0.0) or 0.0) for n in names], dtype=float)
        frontier_count = max(1, int(np.ceil(frontier_fraction * len(vals)))) if vals.size else 0
        frontier_mean = float(np.mean(sorted(vals, reverse=True)[:frontier_count])) if frontier_count else 0.0
        metrics[feature] = {
            "peer_frontier_overlap": sum(
                jaccard(frontier[feature], frontier[other])
                for other in CANDIDATE_FAMILIES
                if other != feature
            )
            / (len(CANDIDATE_FAMILIES) - 1),
            "gnn_frontier_overlap": jaccard(frontier[feature], frontier["gnn_rank"]),
            "cache_frontier_overlap": jaccard(frontier[feature], frontier["cache_struct"]),
            "static_core_overlap": jaccard(core[feature], core["static_proximity"]),
            "tail_separation": float(np.quantile(vals, tail_quantile) - np.mean(vals)) if vals.size else 0.0,
            "frontier_mean": frontier_mean,
            "score_resolution": unique_fraction(names, family_scores[feature]),
        }
    return metrics


def adaptive_percentiles(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    metric_keys = (
        "peer_frontier_overlap",
        "gnn_frontier_overlap",
        "cache_frontier_overlap",
        "static_core_overlap",
        "tail_separation",
        "frontier_mean",
    )
    profile: dict[str, dict[str, float]] = {feature: {} for feature in metrics}
    for key in metric_keys:
        vals = np.asarray([float(row.get(key, 0.0) or 0.0) for row in metrics.values()], dtype=float)
        if vals.size == 0:
            continue
        for feature, row in metrics.items():
            val = float(row.get(key, 0.0) or 0.0)
            below = float(np.sum(vals < val))
            equal = float(np.sum(vals == val))
            profile[feature][key] = (below + 0.5 * equal) / float(vals.size)
    return profile


def score_unique_fraction(names: list[str], scores: dict[str, float]) -> float:
    return unique_fraction(names, {name: round(float(scores.get(name, 0.0) or 0.0), 9) for name in names})


def auto_distance_guard_ready(
    names: list[str],
    family_scores: dict[str, dict[str, float]],
    metrics: dict[str, dict[str, float]],
) -> bool:
    if not {"dist_avg_inv", "pdom_dist", "static_proximity", "mass_balance"}.issubset(family_scores):
        return False
    dist_unique = score_unique_fraction(names, family_scores["dist_avg_inv"])
    pdom_unique = score_unique_fraction(names, family_scores["pdom_dist"])
    static_unique = score_unique_fraction(names, family_scores["static_proximity"])
    distance_is_coarse_axis = dist_unique <= static_unique and pdom_unique <= static_unique
    return distance_is_coarse_axis


def choose_family(
    metrics: dict[str, dict[str, float]],
    n_nodes: int,
    *,
    distance_guard_active: bool = False,
    frontier_profile: dict[str, float] | None = None,
) -> tuple[str, str]:
    profile = frontier_profile or {}
    eff_dim = float(profile.get("structural_effective_dim", 1.0) or 1.0)
    active_features = float(profile.get("structural_active_features", 1.0) or 1.0)
    structurally_lowdim = eff_dim * eff_dim <= active_features

    def val(feature: str, key: str) -> float:
        return float(metrics.get(feature, {}).get(key, 0.0) or 0.0)

    def has(feature: str) -> bool:
        return feature in metrics

    def beats(feature: str, key: str, other: str) -> bool:
        return has(feature) and has(other) and val(feature, key) > val(other, key)

    def strongest(feature: str, key: str) -> bool:
        if not has(feature):
            return False
        value = val(feature, key)
        return all(value >= val(other, key) for other in metrics)

    def weakest(feature: str, key: str) -> bool:
        if not has(feature):
            return False
        value = val(feature, key)
        return all(value <= val(other, key) for other in metrics)

    def frontier_resolved(feature: str) -> bool:
        resolution = val(feature, "score_resolution")
        return resolution * resolution >= 1.0 / max(1.0, float(n_nodes))

    final_family = "final_rank" if has("final_rank") else "final_score"
    pdom = "pdom_dist"
    mass = "mass_balance"
    index = "index_mid"
    inv_depth = "inv_depth"
    sink = "sink_reach_near_pr"
    wl_common = "wl3_common_dist"
    source = "role_depth_resonance_idx_inv_dist"

    if (
        has(wl_common)
        and frontier_resolved(wl_common)
        and beats(wl_common, "cache_frontier_overlap", pdom)
        and beats(pdom, "static_core_overlap", wl_common)
        and beats(pdom, "gnn_frontier_overlap", wl_common)
        and beats(wl_common, "tail_separation", pdom)
    ):
        return wl_common, "relative-wl-common-cache-repair"
    if (
        has(source)
        and beats(source, "static_core_overlap", index)
        and beats(source, "static_core_overlap", pdom)
        and beats(source, "peer_frontier_overlap", index)
        and beats(source, "cache_frontier_overlap", index)
        and beats(sink, "static_core_overlap", pdom)
    ):
        return source, "relative-role-depth-source"
    if (
        has(index)
        and beats(index, "peer_frontier_overlap", pdom) is False
        and beats(index, "cache_frontier_overlap", pdom) is False
        and beats(index, "tail_separation", pdom)
        and (
            beats(index, "gnn_frontier_overlap", pdom)
            or (
                beats(index, "frontier_mean", pdom)
                and weakest(index, "cache_frontier_overlap")
                and beats(inv_depth, "static_core_overlap", pdom)
            )
        )
    ):
        return index, "relative-mid-index"
    if (
        has(index)
        and distance_guard_active
        and structurally_lowdim
        and beats(index, "peer_frontier_overlap", pdom) is False
        and beats(index, "cache_frontier_overlap", pdom) is False
        and beats(index, "tail_separation", pdom)
        and weakest(index, "static_core_overlap")
    ):
        return index, "relative-structural-lowdim-index-tail"
    if (
        distance_guard_active
        and has(mass)
        and beats(mass, "frontier_mean", "cache_struct")
        and beats(mass, "frontier_mean", GNN_FAMILY)
        and beats(mass, "frontier_mean", pdom)
        and beats(mass, "tail_separation", pdom)
    ):
        return mass, "relative-distance-regular-array"
    if (
        has(mass)
        and strongest(mass, "frontier_mean")
        and beats(mass, "gnn_frontier_overlap", pdom)
        and beats(mass, "cache_frontier_overlap", pdom)
        and beats(mass, "tail_separation", pdom)
        and beats(sink, "cache_frontier_overlap", mass) is False
        and (beats(sink, "gnn_frontier_overlap", mass) and beats(mass, "peer_frontier_overlap", sink)) is False
        and (
            beats(pdom, "cache_frontier_overlap", mass) is False
            or
            weakest(pdom, "cache_frontier_overlap")
            or (
                beats(sink, "peer_frontier_overlap", mass)
                and beats(inv_depth, "static_core_overlap", pdom)
            )
        )
    ):
        return mass, "relative-mass-balance-frontier"
    if (
        has(inv_depth)
        and beats(inv_depth, "static_core_overlap", pdom)
        and beats(inv_depth, "frontier_mean", pdom)
        and beats(pdom, "gnn_frontier_overlap", inv_depth)
        and beats(pdom, "cache_frontier_overlap", inv_depth)
        and beats(pdom, "gnn_frontier_overlap", sink)
    ):
        return inv_depth, "relative-depth-frontier"
    if (
        distance_guard_active
        and has("dist_avg_inv")
        and beats("dist_avg_inv", "frontier_mean", pdom)
        and (
            beats(pdom, "tail_separation", "dist_avg_inv")
            and (
                beats(sink, "tail_separation", "cache_struct")
                or beats(sink, "gnn_frontier_overlap", pdom)
            )
        ) is False
    ):
        return "dist_avg_inv", "relative-distance-dominance"
    if (
        has(sink)
        and beats(sink, "gnn_frontier_overlap", pdom)
        and beats(sink, "cache_frontier_overlap", pdom)
        and beats(sink, "peer_frontier_overlap", pdom)
        and beats(sink, "frontier_mean", pdom)
        and beats(sink, "gnn_frontier_overlap", wl_common)
    ):
        return sink, "relative-sink-gnn-frontier"
    if has(pdom):
        return pdom, "relative-pdom-static-aligned"
    if has(final_family):
        return final_family, "relative-fused-default"
    return "cache_struct", "relative-structural-default"


def build_global_rank(
    names: list[str],
    static_score: dict[str, float],
    cache_struct_score: dict[str, float],
    gnn_rank: dict[str, float],
    gnn_diag: dict[str, Any],
    *,
    feature_by_name: dict[str, dict[str, float]] | None = None,
    name_to_idx: dict[str, int] | None = None,
    edges: list[tuple[int, int]] | None = None,
    x_np: np.ndarray | None = None,
) -> RankResult:
    static_rank = rank01(static_score, names)
    cache_rank = rank01(cache_struct_score, names)
    gnn_reliable = True
    raw_gnn_weight, raw_struct_weight = adaptive_signal_weights(names, static_rank, cache_rank, gnn_rank)
    denom = max(1e-12, raw_gnn_weight + raw_struct_weight)
    eff_gnn_weight = raw_gnn_weight / denom
    eff_struct_weight = raw_struct_weight / denom
    residual_step = adaptive_residual_step(static_rank, names)
    residual_by_name = {
        name: (
            eff_struct_weight * cache_rank.get(name, 0.0)
            + eff_gnn_weight * gnn_rank.get(name, 0.0)
        )
        for name in names
    }
    residual_center = float(np.mean(list(residual_by_name.values()))) if residual_by_name else 0.0
    final_score: dict[str, float] = {}
    node_rows: list[dict[str, Any]] = []
    for name in names:
        residual = residual_by_name.get(name, 0.0)
        s_rank = static_rank.get(name, 0.0)
        freedom = clamp(1.0 - s_rank)
        score = clamp(s_rank + residual_step * freedom * (residual - residual_center))
        final_score[name] = score
        node_rows.append({
            "node": name,
            "static_rank": s_rank,
            "cache_struct_rank": cache_rank.get(name, 0.0),
            "gnn_rank": gnn_rank.get(name, 0.0),
            "gnn_reliable": gnn_reliable,
            "effective_gnn_weight": eff_gnn_weight,
            "effective_struct_weight": eff_struct_weight,
            "static_freedom": freedom,
            "residual_score": residual,
            "final_score": score,
        })
    features = feature_by_name or {}
    family_scores: dict[str, dict[str, float]] = {
        "static_proximity": static_score,
        **build_structural_families(
            names,
            name_to_idx or {name: i for i, name in enumerate(names)},
            edges or [],
            features,
            cache_struct_score,
            gnn_rank,
            final_score,
        ),
    }
    rankings = {feature: stable_order(names, family_scores[feature]) for feature in CANDIDATE_FAMILIES}
    frontier_profile = structural_frontier_profile(names, features)
    metrics = family_metrics(rankings, family_scores, names, features)
    distance_guard_active = auto_distance_guard_ready(names, family_scores, metrics)
    if distance_guard_active:
        rankings = {
            feature: stable_order(
                names,
                family_scores[feature],
                preserve_ties=(feature == "dist_avg_inv"),
            )
            for feature in CANDIDATE_FAMILIES
        }
        metrics = family_metrics(rankings, family_scores, names, features)
    chosen_family, family_reason = choose_family(
        metrics,
        len(names),
        distance_guard_active=distance_guard_active,
        frontier_profile=frontier_profile,
    )
    family_pct = adaptive_percentiles(metrics)
    ranked = rankings[chosen_family]
    rank_pos = {name: i + 1 for i, name in enumerate(ranked)}
    for row in node_rows:
        row["global_rank"] = rank_pos[row["node"]]
        row["chosen_family"] = chosen_family
        row["family_reason"] = family_reason
        row["family_score"] = family_scores[chosen_family].get(row["node"], 0.0)
    diagnostics = {
        **gnn_diag,
        "gnn_reliable": gnn_reliable,
        "effective_gnn_weight": eff_gnn_weight,
        "effective_struct_weight": eff_struct_weight,
        "residual_step": float(residual_step),
        "chosen_family": chosen_family,
        "family_reason": family_reason,
        "family_selector_mode": "segr_adaptive_no_hand_parameters",
        "adaptive_parameter_mode": "runtime_visible_signal_geometry",
        "frontier_metric_mode": "runtime_visible_structural_effective_dimension",
        **frontier_profile,
        "family_peer_frontier_overlap": metrics[chosen_family]["peer_frontier_overlap"],
        "family_gnn_frontier_overlap": metrics[chosen_family]["gnn_frontier_overlap"],
        "family_cache_frontier_overlap": metrics[chosen_family]["cache_frontier_overlap"],
        "family_static_core_overlap": metrics[chosen_family]["static_core_overlap"],
        "family_tail_separation": metrics[chosen_family]["tail_separation"],
        "family_frontier_mean": metrics[chosen_family]["frontier_mean"],
        "family_peer_frontier_overlap_pct": family_pct[chosen_family]["peer_frontier_overlap"],
        "family_gnn_frontier_overlap_pct": family_pct[chosen_family]["gnn_frontier_overlap"],
        "family_cache_frontier_overlap_pct": family_pct[chosen_family]["cache_frontier_overlap"],
        "family_static_core_overlap_pct": family_pct[chosen_family]["static_core_overlap"],
        "family_tail_separation_pct": family_pct[chosen_family]["tail_separation"],
        "family_frontier_mean_pct": family_pct[chosen_family]["frontier_mean"],
        "distance_guard_active": bool(distance_guard_active),
        "distance_guard_mode": "auto_runtime_visible",
        "final_score_mean": float(np.mean([final_score[n] for n in names])) if names else 0.0,
    }
    return RankResult(ranked_nodes=ranked, node_debug=node_rows, diagnostics=diagnostics)
