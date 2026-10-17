"""V2-only DDIM adapter with a shared logical start stream and auditable caches.

The update formula is v1 initialization.ddim. The extension accepts explicit
noise/start indices, so a late independent restart uses the same stream as a
large initial population. No original initializer/cache writer is called.
"""
from __future__ import annotations

import json
import numpy as np

from .storage import RESULTS, fingerprint, array_hash, write_arrays, write_json, output_path


def noise_rows(seed, indices, dim):
    return np.asarray([np.random.default_rng(np.random.SeedSequence([int(seed), int(i), 2202]))
                       .standard_normal(dim).astype(np.float32).astype(float) for i in indices]).reshape(-1, dim)


def grid(schedule, steps, image):
    return (np.rint(np.linspace(schedule.T - 1, 0, steps)).astype(int) if image else
            np.unique(np.linspace(schedule.T - 1, 0, steps + 1, dtype=int))[::-1])


def queries_per_start(schedule, steps, image):
    indices = grid(schedule, steps, image)
    return len(indices) if image else len(indices) - 1


def ddim_from_noise(score, noise, steps, image=False):
    x = np.asarray(noise, float).copy()
    indices = grid(score.schedule, steps, image)
    for i, t in enumerate(indices if image else indices[:-1]):
        a = score.schedule.alpha(int(t))
        s = score(x, int(t))
        eps = -np.sqrt(1 - a) * s
        x0 = (x - np.sqrt(1 - a) * eps) / np.sqrt(a)
        if image:
            if not 0 < a < 1:
                raise ValueError("MNIST DDIM requires the trained indexed schedule, including nonclean t=0")
            x0 = np.clip(x0, -1, 1)
            eps = (x - np.sqrt(a) * x0) / np.sqrt(1 - a)
        nxt = int(indices[i + 1]) if i + 1 < len(indices) else -1
        prev = score.schedule.alpha(nxt) if nxt > 0 or image and nxt >= 0 else 1.
        x = np.sqrt(prev) * x0 + np.sqrt(1 - prev) * eps
    return x


class StartFactory:
    def __init__(self, score, seed, steps, identity, image=False, cache=True, prefix_count=None):
        self.score, self.seed, self.steps = score, seed, steps
        self.identity, self.image, self.cache = identity, image, cache
        self.prefix_count = prefix_count
        self.records = []
        self.per_start = queries_per_start(score.schedule, steps, image)

    def __call__(self, indices):
        indices = list(map(int, indices))
        if not indices or min(indices) < 0 or len(set(indices)) != len(indices):
            raise ValueError("Unique nonnegative logical start indices required")
        if self.prefix_count and len(indices) > self.prefix_count and indices[:self.prefix_count] == list(range(self.prefix_count)):
            # Preserve the exact same initial batch/cache for the common P0,
            # even when a sampling baseline buys a much larger pool. Floating
            # point kernels are allowed to depend on batch shape on CPU/MPS.
            return np.vstack([self(indices[:self.prefix_count]), self(indices[self.prefix_count:])])
        payload = dict(format="dms_starts_v2", field=self.identity, seed=self.seed,
                       indices=indices, steps=self.steps, image=self.image, dim=self.score.dim,
                       schedule_alpha_sha256=array_hash(self.score.schedule.alphas_cumprod),
                       clean_zero=getattr(self.score.schedule, "clean_zero", False))
        key = fingerprint(payload)
        path = output_path(RESULTS / "cache" / "starts" / (key + ".npz"))
        meta = path.with_suffix(".json")
        reused = False
        with self.score.cost.scope("initialization"):
            if self.cache and path.is_file() and meta.is_file():
                record = json.loads(meta.read_text())
                with np.load(path, allow_pickle=False) as data:
                    x = data["initial"]
                if record["payload"] != payload or record["array_sha256"] != array_hash(x) or x.shape != (len(indices), self.score.dim):
                    raise ValueError("Corrupt/incompatible v2 initialization cache")
                self.score.cost.charge(len(indices) * self.per_start, self.score.kind, cached=True)
                reused = True
            else:
                x = ddim_from_noise(self.score, noise_rows(self.seed, indices, self.score.dim), self.steps, self.image)
                record = dict(payload=payload, array_sha256=array_hash(x),
                              intrinsic_queries=len(indices) * self.per_start)
                if self.cache:
                    write_arrays(path, initial=x)
                    write_json(meta, record)
        self.records.append(dict(record, reused=reused, cache_key=key))
        return x
