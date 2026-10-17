"""Read-only legacy adapters. Algorithms still execute their original methods."""
from __future__ import annotations

import numpy as np

from research_experiments.common.historical import LegacyProxy, get_class, run_legacy


def corrected_soft_direction(eigenvalues, eigenvectors, threshold=.15):
    """Explicit V2 definition: softness = 1-|lambda|/max|negative lambda|.

    The original all-negative v3f1 branch compares a negative ratio to a positive
    threshold and cannot fire. This NEW ablation changes that branch only.
    """
    negative = np.flatnonzero(eigenvalues < -1e-6)
    if not len(negative):
        return np.empty((0, len(eigenvalues)))
    softness = 1 - np.abs(eigenvalues[negative]) / np.max(np.abs(eigenvalues[negative]))
    j = int(np.argmax(softness))
    return eigenvectors[:, negative[j]][None] if softness[j] > threshold else np.empty((0, len(eigenvalues)))


def split_kernel(score, point, t, cfg, method):
    """Same constructor mapping as v1.split_kernel, with Hessian instrumentation.

    A partial Hessian spends real queries but cannot replace its parent. The
    enclosing v2 coordinator commits an intervention only after it returns.
    """
    kind = "v3f1" if cfg["kind"] == "v3f1_softness_v2" else cfg["kind"]
    cls = get_class(kind)
    proxy = LegacyProxy(score)
    box = (method["box"] if method["projection"] == "clamp" else None) or [-np.inf, np.inf]
    kwargs = dict(timesteps=[t], step_size=method["step_size"] * (np.sqrt(score.dim) if method["step_mode"] == "rms" else 1),
                  split_eps=cfg["delta"], max_split_directions=cfg["max_directions"],
                  hessian_fd_eps=cfg["fd_eps"], ascent_steps=2 * cfg["probe_steps"],
                  normalize_score=method["step_mode"] != "raw", x_min=box[0], x_max=box[1])
    if kind == "v2":
        kwargs.update(split_method="hessian", split_threshold=cfg["absolute_threshold"],
                      hessian_split_eigenvalue_threshold=cfg["eigen_threshold"])
    else:
        kwargs.update(amplification_threshold=cfg["amplification"], tau_abs_min=cfg["absolute_threshold"])
    obj = cls(proxy, **kwargs)
    if cfg["kind"] == "v3f1_softness_v2":
        original_directions = obj._select_split_directions
        obj._select_split_directions = lambda values, vectors: (
            corrected_soft_direction(values, vectors, obj.softness_threshold)
            if not np.any(values > 1e-6) else original_directions(values, vectors))
    original_hessian = obj._approximate_hessian

    def hessian(*args):
        before = score.cost.search_cost
        try:
            with score.cost.scope("split_hessian"):
                return original_hessian(*args)
        finally:
            score.cost.event("finite_difference_score_evaluations", score.cost.search_cost - before)
            score.cost.event("hessian_attempts")

    obj._approximate_hessian = hessian  # instance instrumentation; frozen source is untouched
    with score.cost.scope("split_probes"):
        return obj._check_split_hessian(point.copy(), t)


def original_search(score, initial, method):
    cfg = dict(method, name=method["implementation"])
    with score.cost.scope("original_legacy_search"):
        return run_legacy(score, initial, cfg)
