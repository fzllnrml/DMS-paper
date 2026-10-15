"""Strict configuration and deterministic Cartesian job expansion."""
from __future__ import annotations

from copy import deepcopy
import itertools
import json
from pathlib import Path
import numpy as np
from .storage import REPO, safe_path, fingerprint

METHOD_DEFAULT = dict(name="ascent", levels=[800, 500, 300, 200, 100, 50, 20, 5, 0],
    steps=5, step_size=.1, step_mode="l2", coordinate_mode="indexed",
    final_steps=10, final_scale=.1, final_t=0, tolerance=0., box=None, projection="clamp",
    merge=dict(when="none", radius=.2, radius_factor=None, metric="l2", linkage="complete",
               representative="medoid", every=1, selected=[], stable_threshold=None, direction_cosine=None),
    split=dict(kind="none", every=1, delta=2.5, absolute_threshold=.8, amplification=2.5,
               fd_eps=.001, eigen_threshold=-1., max_directions=2, probe_steps=3,
               max_particles=256, boundary_only=False, boundary_norm=1.),
    selection=dict(kind="none", k=None, bandwidth=1., iterations=20),
    budget_policy="stop", max_extra_starts=0, extra_steps=10, save_every_step=True, historical_kwargs={})

DEFAULT = dict(task="synthetic", suite="01_step_size", phase="development", device="cpu",
    seed=0, distribution_seed=0, model_seed=42, population=32, budget=None,
    distribution=dict(family="moderate", dim=2, components=2, rotate=True, separation=None,
        sigma=1., rare_weight=.01, condition=16., bounds=[-10., 10.]),
    field="exact", initialization=dict(kind="ddim", ddim_steps=50, stop_t=0, source="self", saved_path=None, cache=True),
    method=METHOD_DEFAULT, evaluation=dict(radii=[.1, .25, .5, 1.], reference_restarts=32,
        trajectory_alignment=False, basin_diagnostics=False, stationarity_tolerance=1e-3),
    checkpoint=None, model=dict(T=1000, beta_start=1e-4, beta_end=.02, activation="silu"),
    mnist=dict(evaluator_checkpoint="runs/mnist_evaluator/best.pt", second_evaluator_checkpoint=None,
        data_root="data/mnist", train_limit=None, validation_limit=1024, knn_k=5,
        diagnostic_indices=list(range(16)), stages=True, include_test=False),
    retrospective=dict(roots=["results_pareto_validation_all", "paper_results_archive"],
        manifest=None, radii=[.1, .5, 1., 2., 3., 5.], reductions=[], max_cluster_points=5000))


def merge(a, b):
    out = deepcopy(a)
    for k, v in b.items():
        if k not in a:
            raise ValueError(f"Unknown config key: {k}")
        out[k] = merge(a[k], v) if k != "historical_kwargs" and isinstance(a[k], dict) and isinstance(v, dict) else deepcopy(v)
    return out


def assign(out, path, value):
    parts = path.split(".")
    current = out
    for p in parts[:-1]:
        if p not in current or not isinstance(current[p], dict):
            raise ValueError(f"Invalid grid key {path}")
        current = current[p]
    if parts[-1] not in current:
        raise ValueError(f"Unknown grid key {path}")
    current[parts[-1]] = deepcopy(value)


def input_path(path, overrides=None):
    if path is None:
        return None
    p = Path(path).expanduser()
    overrides = overrides or {}
    if not p.is_absolute():
        for prefix, key in [("runs/", "mnist_root"), ("checkpoints/", "checkpoint_root"), ("data/mnist", "data_root")]:
            if str(p).startswith(prefix) and overrides.get(key):
                suffix = str(p)[len(prefix):].lstrip("/")
                return (Path(overrides[key]).expanduser() / suffix).resolve()
        p = REPO / p
    return p.resolve()


def validate(c):
    if c["task"] not in {"synthetic", "mnist", "retrospective"} or c["phase"] not in {"development", "confirmation"}:
        raise ValueError("Invalid task/phase")
    if c["device"] not in {"cpu", "mps", "auto"}:
        raise ValueError("Only CPU/MPS/auto supported")
    for key in ["seed", "distribution_seed", "model_seed"]:
        if type(c[key]) != int or c[key] < 0:
            raise ValueError(f"Nonnegative integer {key} required")
    if type(c["population"]) != int or c["population"] < 1:
        raise ValueError("Positive population required")
    if c["budget"] is not None and (type(c["budget"]) != int or c["budget"] < 1):
        raise ValueError("Positive integer budget")
    if c["field"] not in {"exact", "learned"}:
        raise ValueError("Invalid field")
    m, d = c["method"], c["distribution"]
    if type(d["dim"]) != int or d["dim"] < 1 or d["components"] < 1:
        raise ValueError("Invalid dimension/components")
    if m["name"] not in {"ascent", "sampling", "historical_v1", "historical_v2", "historical_v3f1", "historical_v3f2", "historical_v3f3"}:
        raise ValueError("Unknown method")
    levels = m["levels"]
    if not levels or levels != sorted(set(levels), reverse=True) or any(type(t) != int or not 0 <= t < c["model"]["T"] for t in levels):
        raise ValueError("Distinct descending indexed levels required")
    if not 0 <= m["final_t"] < c["model"]["T"]:
        raise ValueError("Invalid final_t")
    if isinstance(m["steps"], list):
        if len(m["steps"]) != len(levels) or any(type(n) != int or n < 0 for n in m["steps"]):
            raise ValueError("One nonnegative step count per level")
    elif type(m["steps"]) != int or m["steps"] < 0:
        raise ValueError("Invalid steps")
    sizes = m["step_size"] if isinstance(m["step_size"], list) else [m["step_size"]]
    if len(sizes) not in {1, len(levels)} or any(not np.isfinite(x) or x < 0 for x in sizes):
        raise ValueError("Invalid per-level step sizes")
    if m["step_mode"] not in {"l2", "rms", "raw"} or m["coordinate_mode"] not in {"indexed", "scaled"}:
        raise ValueError("Invalid movement")
    for n in ["final_steps", "extra_steps", "max_extra_starts"]:
        if type(m[n]) != int or m[n] < 0:
            raise ValueError(f"Invalid {n}")
    if m["final_scale"] < 0 or m["tolerance"] < 0:
        raise ValueError("Negative scale/tolerance")
    if m["box"] is not None and (len(m["box"]) != 2 or m["box"][0] >= m["box"][1]):
        raise ValueError("Invalid projection box")
    if m["projection"] not in {"clamp", "none"}:
        raise ValueError("Invalid projection")
    g, s, q = m["merge"], m["split"], m["selection"]
    if g["when"] not in {"none", "final", "every", "selected", "stable"} or g["metric"] not in {"l2", "rms"}:
        raise ValueError("Invalid merge schedule/metric")
    if g["linkage"] not in {"complete", "single", "average"} or g["representative"] not in {"mean", "medoid"}:
        raise ValueError("Invalid merge operator")
    if g["radius"] < 0 or g["every"] < 1 or g["radius_factor"] is not None and g["radius_factor"] < 0:
        raise ValueError("Invalid merge radius/frequency")
    if g["when"] == "stable" and g["stable_threshold"] is None:
        raise ValueError("Stable merge needs a movement threshold")
    if g["direction_cosine"] is not None and not -1 <= g["direction_cosine"] <= 1:
        raise ValueError("Invalid cosine threshold")
    if s["kind"] not in {"none", "v2", "v3f1", "axes"} or s["max_particles"] < 1 or s["every"] < 1:
        raise ValueError("Invalid split configuration")
    if s["fd_eps"] <= 0 or s["delta"] < 0 or s["max_directions"] < 1 or s["probe_steps"] < 1:
        raise ValueError("Invalid split probes")
    if q["kind"] not in {"none", "medoid", "farthest", "facility", "mean_shift"} or q["k"] is not None and q["k"] < 1:
        raise ValueError("Invalid final selection")
    if q["bandwidth"] <= 0 or q["iterations"] < 1:
        raise ValueError("Invalid mean shift")
    if m["budget_policy"] not in {"stop", "continue", "restarts", "farthest_jitter"}:
        raise ValueError("Invalid budget reallocation")
    if m["name"] == "sampling" and m["budget_policy"] != "stop":
        raise ValueError("A sampling baseline must not silently receive score refinement")
    init = c["initialization"]
    if init["source"] not in {"self", "learned"} or init["source"] == "learned" and not c["checkpoint"]:
        raise ValueError("Learned shared initialization needs a checkpoint")
    if init["kind"] not in {"ddim", "sample", "uniform", "ddpm"} or not 1 <= init["ddim_steps"] <= c["model"]["T"]:
        raise ValueError("Invalid initialization")
    if init["saved_path"] and init["kind"] != "ddim":
        raise ValueError("Imported initializations must be DDIM starts")
    if not 0 <= init["stop_t"] < c["model"]["T"]:
        raise ValueError("Invalid stop_t")
    if c["task"] == "mnist" and init["stop_t"] != 0:
        raise ValueError("MNIST shared initialization currently uses a clean DDIM endpoint")
    if m["name"] == "historical_v3f3" and init["stop_t"] != m["historical_kwargs"].get("t_start", max(m["levels"])):
        raise ValueError("Historical v3f3 needs DDIM starts at its t_start")
    if c["task"] == "mnist":
        if c["field"] != "learned" or not c["checkpoint"]:
            raise ValueError("MNIST needs an existing learned checkpoint")
        if m["name"].startswith("historical_"):
            raise ValueError("Historical vector algorithms are restricted to synthetic inputs")
        if init["kind"] not in {"ddim", "ddpm"}:
            raise ValueError("MNIST requires model-generated starts")
    if c["field"] == "learned" and not c["checkpoint"]:
        raise ValueError("No implicit training: learned field needs checkpoint")
    if c["evaluation"]["trajectory_alignment"] and not c["checkpoint"]:
        raise ValueError("Trajectory learned/exact comparison requires an existing checkpoint")
    if init["kind"] == "uniform" and m["box"] is None:
        raise ValueError("Uniform starts need an explicit box")
    if c["field"] == "learned" and init["kind"] == "sample":
        raise ValueError("Use DDIM/uniform for learned score; exact sampling is not a learned baseline")
    if m["name"].startswith("historical_") and (m["coordinate_mode"] != "indexed" or isinstance(m["steps"], list) or isinstance(m["step_size"], list)):
        raise ValueError("Historical adapters require scalar sizes and indexed coordinates")
    if m["name"].startswith("historical_") and m["final_t"] != 0:
        raise ValueError("Historical adapters preserve their original t=0 final refinement")
    if not c["evaluation"]["radii"] or any(x < 0 or not np.isfinite(x) for x in c["evaluation"]["radii"]):
        raise ValueError("Finite evaluation radii required")
    return c


def load(path, overrides=None):
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    if p.suffix == ".json":
        raw = json.loads(p.read_text())
    else:
        import yaml
        raw = yaml.safe_load(p.read_text())
    if set(raw) - {"version", "description", "defaults", "grid", "variants", "output_root"}:
        raise ValueError("Unknown top-level configuration keys")
    if raw.get("version") != 1:
        raise ValueError("Config version must be 1")
    base = merge(DEFAULT, raw.get("defaults", {}))
    jobs = []
    variants = raw.get("variants", [{"label": "default", "config": {}}])
    for variant in variants:
        if set(variant) - {"label", "config", "grid"}:
            raise ValueError("Invalid variant")
        b = merge(base, variant.get("config", {}))
        grid = dict(raw.get("grid", {}), **variant.get("grid", {}))
        keys = sorted(grid)
        if any(not isinstance(grid[k], list) or not grid[k] for k in keys):
            raise ValueError("Nonempty grid axes required")
        for values in itertools.product(*(grid[k] for k in keys)):
            c = deepcopy(b)
            for k, v in zip(keys, values):
                assign(c, k, v)
            validate(c)
            label = variant.get("label", "default")
            if not label or any(x not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for x in label):
                raise ValueError("Variant labels must be filename safe")
            jobs.append(dict(label=label, config=c, id=f"{label}-{fingerprint(c)[:16]}"))
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate resolved jobs")
    output = Path(raw.get("output_root", "research_experiments/results"))
    output = safe_path(output if output.is_absolute() else REPO / output)
    suites = {j["config"]["suite"] for j in jobs}
    if len(suites) != 1:
        raise ValueError("One suite per file")
    digest = fingerprint(dict(jobs=jobs, overrides=overrides or {}))
    return jobs, output, digest
