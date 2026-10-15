"""Paired CPU-generated noise, load-only caches, per-method intrinsic charges."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .storage import fingerprint, array_hash, file_hash, write_arrays, write_json, safe_path


def ddim(score, n, steps, seed, image=False, stop_t=0):
    # Row-major NumPy RNG guarantees common prefixes across different n and
    # batching. Reproduction of old torch seeds is NOT claimed.
    x = np.random.default_rng(seed).standard_normal((n, score.dim)).astype(np.float32).astype(float)
    indices = (np.rint(np.linspace(score.schedule.T - 1, 0, steps)).astype(int) if image else
               np.unique(np.linspace(score.schedule.T - 1, stop_t, steps + 1, dtype=int))[::-1])
    evaluation_indices = indices if image else indices[:-1]
    for i, t in enumerate(evaluation_indices):
        a = score.schedule.alpha(int(t))
        s = score(x, int(t))
        eps = -np.sqrt(1 - a) * s
        x0 = (x - np.sqrt(1 - a) * eps) / np.sqrt(a)
        if image:
            x0 = np.clip(x0, -1, 1)
            eps = (x - np.sqrt(a) * x0) / np.sqrt(1 - a)
        nxt = int(indices[i + 1]) if i + 1 < len(indices) else -1
        prev = score.schedule.alpha(nxt) if nxt > 0 or image and nxt >= 0 else 1.
        x = np.sqrt(prev) * x0 + np.sqrt(1 - prev) * eps
    return x


def ddpm(score, n, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, score.dim))
    for t in reversed(range(score.schedule.T)):
        beta = score.schedule.betas[t]
        s = score(x, t)
        # Matches BaselineModeFinder._sample_reverse variance convention.
        x = (x + beta * s) / np.sqrt(1 - beta)
        if t:
            x += np.sqrt(beta) * rng.standard_normal(x.shape)
    return x


def initialize(score, c, cache_root, identity, mixture=None, n=None, seed=None, external_start=None):
    n, seed = n or c["population"], c["seed"] if seed is None else seed
    init = c["initialization"]
    payload = dict(version=1, field_identity=identity, seed=seed, n=n, dim=score.dim,
                   initialization=init, image=c["task"] == "mnist")
    if external_start:
        payload["external_sha256"] = file_hash(external_start)
    key = fingerprint(payload)
    cache = safe_path(Path(cache_root) / (key + ".npz"))
    meta = cache.with_suffix(".json")
    if init["cache"] and cache.exists() and meta.exists():
        info = json.loads(meta.read_text())
        if info["payload"] != payload:
            raise ValueError("Initialization cache identity mismatch")
        with np.load(cache, allow_pickle=False) as data:
            x = data["initial"]
        if array_hash(x) != info["array_hash"]:
            raise ValueError("Corrupt start cache")
        score.cost.charge(info["cost"], score.kind, cached=True)
        return x, dict(info, reused=True)
    before = score.cost.search_cost
    if external_start:
        external_start = Path(external_start)
        source = json.loads(external_start.with_suffix(".json").read_text())
        if source.get("format") != "dms_saved_starts_v1" or source["seed"] != seed or source["ddim_steps"] != init["ddim_steps"]:
            raise ValueError("Saved start protocol/seed/steps mismatch")
        if score.kind != "neural" or source["checkpoint_sha256"] != identity.get("checkpoint_sha256"):
            raise ValueError("Saved starts do not belong to the requested neural initialization checkpoint")
        with np.load(external_start, allow_pickle=False) as data:
            cloud = data["initial"]
        if array_hash(cloud) != source["array_hash"] or cloud.ndim != 2 or cloud.shape[1] != score.dim or len(cloud) < n:
            raise ValueError("Invalid saved start array/prefix")
        x = cloud[:n].copy()
        per_point = source["initialization_nfe"] / len(cloud)
        if int(per_point) != per_point:
            raise ValueError("Initialization NFE is not a whole per-point charge")
        score.cost.charge(int(per_point) * n, "neural", cached=True)
    elif init["kind"] == "ddim":
        x = ddim(score, n, init["ddim_steps"], seed, c["task"] == "mnist", init["stop_t"])
    elif init["kind"] == "ddpm":
        x = ddpm(score, n, seed)
    elif init["kind"] == "sample":
        if mixture is None:
            raise ValueError("Exact distribution sampling requires a synthetic mixture")
        x = mixture.sample(n, seed)
    elif init["kind"] == "uniform":
        box = c["method"]["box"]
        if box is None:
            raise ValueError("Uniform starts require an explicit box")
        x = np.random.default_rng(seed).uniform(*box, (n, score.dim))
    else:
        raise ValueError(init["kind"])
    info = dict(payload=payload, cost=score.cost.search_cost - before,
                array_hash=array_hash(x), reused=bool(external_start))
    if init["cache"]:
        write_arrays(cache, initial=x)
        write_json(meta, info)
    return x, info
