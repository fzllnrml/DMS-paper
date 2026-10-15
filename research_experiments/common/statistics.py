"""Cluster-aware uncertainty for runs, never treating particles as replicates."""
from __future__ import annotations

import numpy as np


def hierarchical_interval(values_by_cluster, seed=991, repeats=2000, alpha=.05):
    groups = [np.asarray(g, float) for g in values_by_cluster if len(g)]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    if not groups:
        return dict(mean=None, low=None, high=None, n_clusters=0, n_runs=0)
    # Equal weight per independent mixture/training, then equal weight per seed.
    observed = float(np.mean([g.mean() for g in groups]))
    if sum(map(len, groups)) == 1:
        return dict(mean=observed, low=None, high=None, n_clusters=1, n_runs=1,
                    scope="single run: no empirical uncertainty interval")
    rng, estimates = np.random.default_rng(seed), []
    for _ in range(repeats):
        selected = rng.integers(0, len(groups), len(groups)) if len(groups) > 1 else [0]
        estimates.append(np.mean([rng.choice(groups[i], len(groups[i]), replace=True).mean() for i in selected]))
    low, high = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return dict(mean=observed, low=float(low), high=float(high), n_clusters=len(groups),
        n_runs=sum(map(len, groups)), repeats=repeats, alpha=alpha,
        scope="mixture/model then search seed" if len(groups) > 1 else "conditional on one fixed mixture/model",
        extreme_tail_expected_draws=repeats * alpha / 2)
