"""Evaluation only. No method imports this module or receives reference modes."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def points(x, dim=None):
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return np.empty((0, dim or (a.shape[1] if a.ndim == 2 else 0)))
    if a.ndim == 1:
        a = a.reshape(-1, 1) if dim == 1 else a.reshape(1, -1)
    if a.ndim != 2 or (dim is not None and a.shape[1] != dim):
        raise ValueError("Expected [N, dim] points")
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite coordinates")
    return a


def one_to_one(distances, radius):
    """Maximum cardinality inside radius, then minimum total matched distance.

    Each true mode receives a unique real candidate or its own dummy column.
    One extra real match saves more than all possible distance tie breakers.
    Invalid edges cost more than a dummy and can never be selected.
    """
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or radius < 0 or not np.isfinite(radius):
        raise ValueError("Invalid distance matrix/radius")
    k, m = d.shape
    if not k or not m:
        return np.empty((0, 2), dtype=int)
    penalty = float(min(k, m) + 1)
    scaled = d / max(radius, 1e-300)
    cost = np.full((k, m + k), penalty)
    cost[:, :m] = np.where(d <= radius, np.minimum(scaled, 1), 2 * penalty)
    rows, cols = linear_sum_assignment(cost)
    good = (cols < m)
    rows, cols = rows[good], cols[good]
    good = d[rows, cols] <= radius
    return np.column_stack((rows[good], cols[good]))


def evaluate_modes(reference, candidates, radii, rare_mask=None):
    ref = points(reference)
    cand = points(candidates, ref.shape[1])
    k, m = len(ref), len(cand)
    d = cdist(ref, cand)
    nearest = d.min(1) if m else np.full(k, np.inf)
    correct_dist = d.min(0) if k else np.full(m, np.inf)
    rare = np.zeros(k, bool) if rare_mask is None else np.asarray(rare_mask, bool)
    if rare.shape != (k,):
        raise ValueError("rare_mask shape")
    outputs = []
    for radius in radii:
        pairs = one_to_one(d, float(radius))
        hits, correct = nearest <= radius, correct_dist <= radius
        r = float(hits.mean()) if k else None
        p = (float(correct.mean()) if m else 0.0) if k else None
        r1, p1 = (len(pairs) / k if k else None), ((len(pairs) / m if m else 0.0) if k else None)
        def f1(a, b):
            if a is None or b is None:
                return None
            return 2 * a * b / (a + b) if a + b else 0.0
        # Duplicates here are excess valid candidates beyond maximum matching.
        # Invalid candidates are false positives, not automatically duplicates.
        duplicate_count = int(correct.sum() - len(pairs))
        outputs.append(dict(radius=float(radius), recall=r, precision=p,
            f1=f1(r, p), one_to_one_recall=r1, one_to_one_precision=p1,
            one_to_one_f1=f1(r1, p1), n_reference=k, n_candidates=m,
            n_hits=int(hits.sum()), n_matched=len(pairs),
            duplicate_count=duplicate_count,
            duplicate_fraction=duplicate_count / m if m else 0.0,
            soft_error=float(nearest.mean()) if k else None,
            localization_error=float(nearest[hits].mean()) if hits.any() else None,
            matched_localization_error=float(d[pairs[:, 0], pairs[:, 1]].mean()) if len(pairs) else None,
            rare_recall=float(hits[rare].mean()) if rare.any() else None,
            per_mode_distance=nearest.tolist(), matched_pairs=pairs.tolist()))
    return outputs


def support_percentile(values, reference):
    r = np.sort(np.asarray(reference, float))
    if r.size == 0 or not np.isfinite(r).all():
        raise ValueError("Finite nonempty calibration required")
    v = np.asarray(values, float)
    return 100 - 50 * (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / len(r)


def pareto_mask(values, minimize):
    """Nondomination without weighting metrics; identical points both survive."""
    v = np.asarray(values, float).copy()
    v *= np.where(np.asarray(minimize), 1, -1)
    if not np.isfinite(v).all():
        raise ValueError("Pareto requires finite values")
    return np.array([not np.any(np.all(v <= x, axis=1) & np.any(v < x, axis=1)) for x in v])
