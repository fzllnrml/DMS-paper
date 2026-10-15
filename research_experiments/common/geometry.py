"""Geometry-only operations; no score, reference modes or classifier access."""
from __future__ import annotations

import math
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, cut_tree
from scipy.spatial.distance import cdist, squareform


def step_delta(score, length, mode="l2"):
    s = np.asarray(score, float)
    if s.ndim != 2 or not np.isfinite(s).all() or not np.isfinite(length) or length < 0:
        raise ValueError("Invalid score/length")
    if mode == "raw":
        return length * s
    if mode not in {"l2", "rms"}:
        raise ValueError("Invalid step mode")
    norm = np.linalg.norm(s, axis=1, keepdims=True)
    direction = s / np.where(norm > 1e-8, norm, 1.)
    return length * direction * (math.sqrt(s.shape[1]) if mode == "rms" else 1.)


def farthest_indices(x, k):
    n = len(x)
    if k < 1:
        return np.array([], int)
    if n <= k:
        return np.arange(n)
    selected, nearest = [0], np.full(n, np.inf)
    while len(selected) < k:
        nearest = np.minimum(nearest, np.linalg.norm(x - x[selected[-1]], axis=1))
        nearest[selected] = -np.inf
        selected.append(int(nearest.argmax()))
    return np.array(selected, int)


def cluster(x, radius, representative="medoid", metric="l2", method="complete", k=None, directions=None, cosine_threshold=None):
    x = np.asarray(x, float)
    if x.ndim != 2 or radius < 0:
        raise ValueError("Invalid cluster input")
    if not len(x):
        return x.copy(), [], []
    d = cdist(x, x) / (math.sqrt(x.shape[1]) if metric == "rms" else 1.)
    if cosine_threshold is not None:
        if method != "complete" or k is not None:
            raise ValueError("Direction gate requires complete linkage and a radius cut")
        v = np.asarray(directions, float)
        u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        incompatible = u @ u.T < cosine_threshold
        d[incompatible] = max(float(d.max()) + 1, radius + 1)
        np.fill_diagonal(d, 0)
    if len(x) < 2:
        labels = np.zeros(len(x), int)
    elif k is not None:
        if not 1 <= k <= len(x):
            raise ValueError("Fixed K must not exceed candidate count")
        z = linkage(squareform(d, checks=False), method=method)
        labels = cut_tree(z, n_clusters=[k]).ravel()
    elif radius == 0:
        # Exact duplicate removal (unlike historical radius=0 meaning no merge).
        _, labels = np.unique(x, axis=0, return_inverse=True)
    else:
        z = linkage(squareform(d, checks=False), method=method)
        labels = fcluster(z, t=radius, criterion="distance") - 1
    groups = sorted((np.flatnonzero(labels == v).tolist() for v in np.unique(labels)), key=lambda ids: ids[0])
    outputs, reps = [], []
    for ids in groups:
        if representative == "mean":
            outputs.append(x[ids].mean(0))
            reps.append(None)
        elif representative == "medoid":
            j = ids[int(np.argmin(d[np.ix_(ids, ids)].sum(1)))]
            outputs.append(x[j])
            reps.append(j)
        else:
            raise ValueError("Unknown representative")
    return np.array(outputs), groups, reps


def select(x, cfg):
    """Fixed K is an output constraint, never selected against reference modes."""
    x = np.asarray(x, float)
    if len(x) == 0 or cfg["kind"] == "none":
        return x.copy()
    k = min(cfg["k"] or len(x), len(x))
    if cfg["kind"] == "farthest":
        return x[farthest_indices(x, k)]
    if cfg["kind"] == "medoid":
        return cluster(x, 0, k=k)[0]
    if cfg["kind"] == "facility":
        dist = cdist(x, x)
        selected, best = [], np.full(len(x), np.inf)
        for _ in range(k):
            costs = np.minimum(best[:, None], dist).sum(0)
            costs[selected] = np.inf
            idx = int(np.argmin(costs))
            selected.append(idx)
            best = np.minimum(best, dist[:, idx])
        return x[selected]
    if cfg["kind"] == "mean_shift":
        # Fixed KDE on the original cloud; update ALL centers simultaneously.
        source, y = x.copy(), x.copy()
        for _ in range(cfg["iterations"]):
            logw = -cdist(y, source, "sqeuclidean") / (2 * cfg["bandwidth"] ** 2)
            w = np.exp(logw - logw.max(1, keepdims=True))
            y = (w @ source) / w.sum(1, keepdims=True)
        return cluster(y, cfg["bandwidth"] * .1, k=k if cfg["k"] else None)[0]
    raise ValueError(cfg["kind"])
