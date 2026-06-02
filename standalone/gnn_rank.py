from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .data_io import edge_index_tensor, minmax, rank01


@dataclass
class GnnResult:
    score: dict[str, float]
    rank: dict[str, float]
    diagnostics: dict[str, Any]


class GcnEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList()
        last = in_dim
        for _ in range(max(1, int(layers))):
            self.layers.append(nn.Linear(last, hidden))
            last = hidden
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = torch.sparse.mm(adj, h)
            h = layer(h)
            h = F.prelu(h, torch.tensor(0.25, device=h.device))
            if self.dropout > 0:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GcnDenoisingAutoEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.encoder = GcnEncoder(in_dim, hidden, layers, dropout)
        self.decoder = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x, adj)
        return self.decoder(z), z


def normalized_adj(n: int, edges: list[tuple[int, int]], device: torch.device) -> torch.Tensor:
    pairs = [(i, i) for i in range(n)]
    pairs.extend((int(u), int(v)) for u, v in edges if 0 <= int(u) < n and 0 <= int(v) < n)
    pairs.extend((int(v), int(u)) for u, v in edges if 0 <= int(u) < n and 0 <= int(v) < n)
    idx = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
    vals = torch.ones(idx.size(1), dtype=torch.float32, device=device)
    deg = torch.zeros(n, dtype=torch.float32, device=device)
    deg.scatter_add_(0, idx[0], vals)
    norm = vals / torch.sqrt(deg[idx[0]].clamp_min(1.0) * deg[idx[1]].clamp_min(1.0))
    return torch.sparse_coo_tensor(idx, norm, (n, n), device=device).coalesce()


def choose_train_nodes(n: int, max_nodes: int, seed: int) -> np.ndarray:
    if n <= max_nodes:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=max_nodes, replace=False).astype(np.int64))


def induced_edges(edges: list[tuple[int, int]], nodes: np.ndarray) -> tuple[list[tuple[int, int]], dict[int, int]]:
    node_set = set(int(x) for x in nodes.tolist())
    mapping = {int(old): i for i, old in enumerate(nodes.tolist())}
    out: list[tuple[int, int]] = []
    for u, v in edges:
        if int(u) in node_set and int(v) in node_set:
            out.append((mapping[int(u)], mapping[int(v)]))
    return out, mapping


def score_gnn(
    names: list[str],
    name_to_idx: dict[str, int],
    x_np: np.ndarray,
    edges: list[tuple[int, int]],
    cache_struct_score: dict[str, float],
    *,
    epochs: int = 30,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.10,
    device: str = "cuda",
    train_node_cap: int = 8192,
    seed: int = 123,
) -> GnnResult:
    n = int(x_np.shape[0])
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_nodes = choose_train_nodes(n, int(train_node_cap), seed)
    train_edges, mapping = induced_edges(edges, train_nodes)
    x_train_np = x_np[train_nodes].astype(np.float32, copy=False)
    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=dev)
    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mu) / sigma
    adj_train = normalized_adj(len(train_nodes), train_edges, dev)
    model = GcnEncoder(x_train.size(1), int(hidden), int(layers), float(dropout)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(max(1, int(epochs))):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model(x_train, adj_train)
        perm = torch.randperm(x_train.size(0), device=dev)
        z_bad = model(x_train[perm], adj_train)
        summary = torch.sigmoid(z.mean(dim=0))
        pos = torch.sum(z * summary, dim=1)
        neg = torch.sum(z_bad * summary, dim=1)
        loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        loss = loss + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        loss.backward()
        opt.step()
    model.eval()
    x_full = torch.tensor(x_np.astype(np.float32, copy=False), dtype=torch.float32, device=dev)
    x_full = (x_full - mu) / sigma
    adj_full = normalized_adj(n, edges, dev)
    with torch.no_grad():
        z_full = model(x_full, adj_full).detach().cpu().numpy()
    emb_norm = minmax(np.linalg.norm(z_full, axis=1))
    score = {name: float(emb_norm[name_to_idx[name]]) for name in names}
    rank = rank01(score, names)
    random_values = {name: float(((name_to_idx[name] * 1103515245 + seed) % 1000003) / 1000003.0) for name in names}
    random_rank = rank01(random_values, names)
    struct_rank = rank01(cache_struct_score, names)
    top_n = max(1, min(64, len(names)))
    top_gnn = sorted(names, key=lambda nm: rank.get(nm, 0.0), reverse=True)[:top_n]
    gnn_nonrandom = 0.5 + 0.5 * float(np.mean([rank.get(nm, 0.0) - random_rank.get(nm, 0.0) for nm in top_gnn]))
    gnn_struct_agree = float(np.mean([min(rank.get(nm, 0.0), struct_rank.get(nm, 0.0)) for nm in top_gnn]))
    diagnostics = {
        "gnn_nonrandom": max(0.0, min(1.0, gnn_nonrandom)),
        "gnn_struct_agree": max(0.0, min(1.0, gnn_struct_agree)),
        "train_nodes": int(len(train_nodes)),
        "device": str(dev),
        "epochs": int(epochs),
        "hidden": int(hidden),
        "layers": int(layers),
    }
    return GnnResult(score=score, rank=rank, diagnostics=diagnostics)


def score_gnn_denoising(
    names: list[str],
    name_to_idx: dict[str, int],
    x_np: np.ndarray,
    edges: list[tuple[int, int]],
    cache_struct_score: dict[str, float],
    *,
    epochs: int = 30,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.10,
    device: str = "cuda",
    train_node_cap: int = 8192,
    seed: int = 123,
    mask_prob: float = 0.15,
) -> GnnResult:
    del cache_struct_score
    n = int(x_np.shape[0])
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_nodes = choose_train_nodes(n, int(train_node_cap), seed)
    train_edges, _mapping = induced_edges(edges, train_nodes)
    x_train_np = x_np[train_nodes].astype(np.float32, copy=False)
    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=dev)
    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mu) / sigma
    adj_train = normalized_adj(len(train_nodes), train_edges, dev)
    model = GcnDenoisingAutoEncoder(x_train.size(1), int(hidden), int(layers), float(dropout)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    mask_p = max(0.0, min(0.95, float(mask_prob)))
    for _ in range(max(1, int(epochs))):
        model.train()
        opt.zero_grad(set_to_none=True)
        keep = (torch.rand_like(x_train) >= mask_p).float()
        recon, _z = model(x_train * keep, adj_train)
        loss = F.mse_loss(recon, x_train)
        loss.backward()
        opt.step()
    model.eval()
    x_full = torch.tensor(x_np.astype(np.float32, copy=False), dtype=torch.float32, device=dev)
    x_full = (x_full - mu) / sigma
    adj_full = normalized_adj(n, edges, dev)
    with torch.no_grad():
        recon_full, z_full = model(x_full, adj_full)
        err = torch.mean((recon_full - x_full) ** 2, dim=1).detach().cpu().numpy()
        z_np = z_full.detach().cpu().numpy()
    score = {name: float(err[name_to_idx[name]]) for name in names}
    rank = rank01(score, names)
    diagnostics = {
        "gnn_objective": "denoising_feature_reconstruction",
        "train_nodes": int(len(train_nodes)),
        "device": str(dev),
        "epochs": int(epochs),
        "hidden": int(hidden),
        "layers": int(layers),
        "mask_prob": mask_p,
        "embedding_norm_mean": float(np.mean(np.linalg.norm(z_np, axis=1))) if z_np.size else 0.0,
    }
    return GnnResult(score=score, rank=rank, diagnostics=diagnostics)


def _sample_negative_edges(num_nodes: int, positives: set[tuple[int, int]], count: int, dev: torch.device) -> torch.Tensor:
    if num_nodes <= 1 or count <= 0:
        return torch.empty((2, 0), dtype=torch.long, device=dev)
    out: list[tuple[int, int]] = []
    attempts = 0
    max_attempts = max(100, count * 20)
    while len(out) < count and attempts < max_attempts:
        attempts += 1
        u = int(np.random.randint(0, num_nodes))
        v = int(np.random.randint(0, num_nodes))
        if u == v:
            continue
        key = (min(u, v), max(u, v))
        if key in positives:
            continue
        out.append((u, v))
    if not out:
        return torch.empty((2, 0), dtype=torch.long, device=dev)
    return torch.tensor(out, dtype=torch.long, device=dev).t().contiguous()


def score_gnn_link_prediction(
    names: list[str],
    name_to_idx: dict[str, int],
    x_np: np.ndarray,
    edges: list[tuple[int, int]],
    cache_struct_score: dict[str, float],
    *,
    epochs: int = 30,
    hidden: int = 64,
    layers: int = 2,
    dropout: float = 0.10,
    device: str = "cuda",
    train_node_cap: int = 8192,
    seed: int = 123,
) -> GnnResult:
    del cache_struct_score
    n = int(x_np.shape[0])
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_nodes = choose_train_nodes(n, int(train_node_cap), seed)
    train_edges, mapping = induced_edges(edges, train_nodes)
    x_train_np = x_np[train_nodes].astype(np.float32, copy=False)
    x_train = torch.tensor(x_train_np, dtype=torch.float32, device=dev)
    mu = x_train.mean(dim=0, keepdim=True)
    sigma = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mu) / sigma
    adj_train = normalized_adj(len(train_nodes), train_edges, dev)
    model = GcnEncoder(x_train.size(1), int(hidden), int(layers), float(dropout)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    pos_pairs = sorted({(min(int(u), int(v)), max(int(u), int(v))) for u, v in train_edges if int(u) != int(v)})
    if not pos_pairs:
        score = {name: 0.0 for name in names}
        return GnnResult(score=score, rank=rank01(score, names), diagnostics={"gnn_objective": "link_prediction", "train_edges": 0})
    pos_idx = torch.tensor(pos_pairs, dtype=torch.long, device=dev).t().contiguous()
    positives = set(pos_pairs)
    for _ in range(max(1, int(epochs))):
        model.train()
        opt.zero_grad(set_to_none=True)
        z = model(x_train, adj_train)
        neg_idx = _sample_negative_edges(len(train_nodes), positives, pos_idx.size(1), dev)
        pos_logits = torch.sum(z[pos_idx[0]] * z[pos_idx[1]], dim=1)
        if neg_idx.numel() > 0:
            neg_logits = torch.sum(z[neg_idx[0]] * z[neg_idx[1]], dim=1)
            logits = torch.cat([pos_logits, neg_logits])
            labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)])
        else:
            logits = pos_logits
            labels = torch.ones_like(pos_logits)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()
    model.eval()
    x_full = torch.tensor(x_np.astype(np.float32, copy=False), dtype=torch.float32, device=dev)
    x_full = (x_full - mu) / sigma
    adj_full = normalized_adj(n, edges, dev)
    with torch.no_grad():
        z_full = model(x_full, adj_full).detach().cpu().numpy()
    emb_norm = minmax(np.linalg.norm(z_full, axis=1))
    score = {name: float(emb_norm[name_to_idx[name]]) for name in names}
    rank = rank01(score, names)
    diagnostics = {
        "gnn_objective": "link_prediction",
        "train_nodes": int(len(train_nodes)),
        "train_edges": int(len(pos_pairs)),
        "device": str(dev),
        "epochs": int(epochs),
        "hidden": int(hidden),
        "layers": int(layers),
    }
    return GnnResult(score=score, rank=rank, diagnostics=diagnostics)
