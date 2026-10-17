"""Strict, small declarative protocols; dry-run never loads a neural model."""
from __future__ import annotations

from copy import deepcopy
import itertools
import json
from pathlib import Path

import numpy as np

from research_experiments.common.config import METHOD_DEFAULT, merge, assign
from .storage import ROOT, REPO, RESULTS, fingerprint, input_path

METHOD = dict(deepcopy(METHOD_DEFAULT), implementation="population", refill="refine",
              stable_window=3, stable_score_max=.1, split_levels=[100, 20, 0])
for _key in ["selection", "budget_policy", "max_extra_starts", "extra_steps"]:
    METHOD.pop(_key)
for _key in ["every", "boundary_only", "boundary_norm"]:
    METHOD["split"].pop(_key)
METHOD.update(step_size=.3, steps=5, final_steps=10, final_scale=.1, save_every_step=False)
DEFAULT = dict(domain="exact", field="exact", phase="pilot", study="mechanisms", device="auto", seed=0,
               distribution_seed=0, model_seed=None, population=8,
               problem=dict(family="moderate", dim=10, components=2, rare_weight=.003),
               budget=dict(limit=2048, regime="matched", tolerance=0),
               initialization=dict(steps=20, source="self", cache=True), method=METHOD,
               outputs=[dict(name="native", kind="radius", k=None, radius=.25, metric="l2", representative="medoid"),
                        dict(name="le8", kind="at_most", k=8, radius=.25, metric="l2", representative="medoid"),
                        dict(name="eq8", kind="exact", k=8, radius=.25, metric="l2", representative="medoid"),
                        dict(name="eq8_mean", kind="exact", k=8, radius=.25, metric="l2", representative="mean")],
               checkpoint=None, model=dict(T=1000, beta_start=1e-4, beta_end=.02, activation="silu"),
               evaluation=dict(radii=[.1, .25, .5, 1.], alignment=False, basin_diagnostics=True,
                               diagnostic_points=16),
               mnist=dict(evaluator_checkpoint="runs/mnist_evaluator/best.pt", data_root="data/mnist",
                          train_limit=None, validation_limit=1024, knn_k=5, train_close_quantile=.01,
                          near_duplicate_rms=.02))


def validate(c):
    if c["domain"] not in {"exact", "alignment", "mnist"}:
        raise ValueError("Unknown domain")
    if c["field"] not in {"exact", "learned"} or c["domain"] == "exact" and c["field"] != "exact" or c["domain"] == "mnist" and c["field"] != "learned":
        raise ValueError("Field/domain mismatch")
    if c["phase"] not in {"pilot", "pilot_tune", "pilot_transfer", "confirmation"}:
        raise ValueError("Unknown phase")
    if c["device"] not in {"cpu", "mps", "auto"}:
        raise ValueError("CPU/MPS/auto required")
    for k in ["seed", "distribution_seed", "model_seed"]:
        if k == "model_seed" and c["domain"] != "mnist" and c[k] is None:
            continue
        if type(c[k]) != int or c[k] < 0:
            raise ValueError(f"Invalid {k}")
    if type(c["population"]) != int or c["population"] < 1:
        raise ValueError("Positive population required")
    b, m = c["budget"], c["method"]
    if set(m) != set(METHOD) or set(m["split"]) != set(METHOD["split"]) or set(m["merge"]) != set(METHOD["merge"]):
        raise ValueError("Unknown/missing method keys; V2 owns refill/output policies")
    if (type(b["limit"]) != int or b["limit"] < 1 or b["regime"] not in {"cap", "matched"}
            or type(b["tolerance"]) != int or b["tolerance"] < 0):
        raise ValueError("Invalid NFE protocol")
    if m["implementation"] not in {"population", "historical_v2", "historical_v3f1", "historical_v3f2"}:
        raise ValueError("Unknown implementation")
    if m["name"] not in {"ascent", "sampling"} or m["refill"] not in {"none", "refine", "restarts"}:
        raise ValueError("Unknown method/policy")
    levels = m["levels"]
    if not levels or levels != sorted(set(levels), reverse=True) or any(type(t) != int or not 0 <= t < c["model"]["T"] for t in levels):
        raise ValueError("Distinct descending indexed levels required")
    for key in ["steps", "final_steps", "stable_window"]:
        values = m[key] if isinstance(m[key], list) else [m[key]]
        if any(type(v) != int or v < (1 if key == "stable_window" else 0) for v in values):
            raise ValueError(f"Invalid {key}")
        if isinstance(m[key], list) and (key != "steps" or len(values) != len(levels)):
            raise ValueError("Per-level steps length mismatch")
    sizes = m["step_size"] if isinstance(m["step_size"], list) else [m["step_size"]]
    if len(sizes) not in {1, len(levels)} or any(not np.isfinite(s) or s < 0 for s in sizes):
        raise ValueError("Invalid step sizes")
    if m["step_mode"] not in {"raw", "l2", "rms"} or m["coordinate_mode"] != "indexed":
        raise ValueError("V2 primary protocols use declared indexed coordinates")
    if m["projection"] not in {"clamp", "none"} or m["final_t"] != 0 or m["final_scale"] < 0:
        raise ValueError("Invalid final movement/projection")
    if m["box"] is not None and (len(m["box"]) != 2 or m["box"][0] >= m["box"][1]):
        raise ValueError("Invalid box")
    g, s = m["merge"], m["split"]
    if g["when"] not in {"none", "every", "selected", "stable", "final"}:
        raise ValueError("Invalid merge schedule")
    if (not np.isfinite(g["radius"]) or g["radius"] < 0 or g["every"] < 1 or g["metric"] not in {"l2", "rms"}
            or g["linkage"] != "complete" or g["representative"] not in {"mean", "medoid"}
            or g["radius_factor"] is not None and g["radius_factor"] < 0):
        raise ValueError("Invalid complete-linkage merge")
    if g["when"] == "stable" and g["stable_threshold"] is None:
        raise ValueError("Stable gate needs a window displacement threshold")
    if s["kind"] not in {"none", "v2", "v3f1", "v3f1_softness_v2"} or s["fd_eps"] <= 0 or s["max_particles"] < c["population"]:
        raise ValueError("Invalid Split configuration")
    if m["tolerance"] != 0:
        raise ValueError("Primary V2 movement has fixed step counts; use explicit legacy controls for original stopping")
    if m["name"] == "sampling" and (m["refill"] != "none" or g["when"] != "none" or s["kind"] != "none"):
        raise ValueError("Sampling control cannot receive hidden ascent/Merge/Split")
    if m["implementation"].startswith("historical_") and (m["refill"] != "none" or isinstance(m["step_size"], list) or isinstance(m["steps"], list)):
        raise ValueError("Original stopping rules/scalar parameters required for historical controls")
    init = c["initialization"]
    if not 2 <= init["steps"] <= c["model"]["T"] or init["source"] not in {"self", "learned"}:
        raise ValueError("Invalid DDIM initialization")
    per_start = init["steps"] if c["domain"] == "mnist" else min(init["steps"], c["model"]["T"] - 1)
    if b["limit"] < c["population"] * per_start:
        raise ValueError("Budget cannot afford the declared common initial population")
    if c["domain"] == "exact" and (c["checkpoint"] or init["source"] != "self"):
        raise ValueError("Primary exact benchmark must not depend on neural models")
    if c["domain"] in {"alignment", "mnist"} and not c["checkpoint"]:
        raise ValueError("Existing checkpoint required; implicit training is forbidden")
    if c["domain"] == "alignment" and c["problem"]["family"] != "legacy_alignment":
        raise ValueError("Saved synthetic models belong to the legacy_random density only")
    if c["domain"] == "alignment" and init["source"] != "learned":
        raise ValueError("Exact/learned pairs require the same learned DDIM starts")
    if c["domain"] == "mnist" and c["problem"]["dim"] != 784:
        raise ValueError("MNIST coordinates are 784 normalized pixels in [-1,1]")
    if c["domain"] == "mnist" and m["implementation"] != "population":
        raise ValueError("Original full vector-class controls are restricted to synthetic inputs")
    if type(c["problem"]["dim"]) != int or c["problem"]["dim"] < 1:
        raise ValueError("Positive dimension required")
    if c["domain"] == "exact" and c["problem"]["family"] not in {"close", "overlap", "rare", "many", "anisotropic", "uneven", "sheared", "moderate", "collinear_gmm"}:
        raise ValueError("Unknown primary exact density family")
    for out in c["outputs"]:
        if set(out) != {"name", "kind", "k", "radius", "metric", "representative"}:
            raise ValueError("Output schema mismatch")
        if out["kind"] not in {"radius", "exact", "at_most"} or out["radius"] < 0:
            raise ValueError("Invalid output constraint")
        if out["kind"] != "radius" and (type(out["k"]) != int or out["k"] < 1):
            raise ValueError("K required")
        if out["metric"] not in {"l2", "rms"} or out["representative"] not in {"mean", "medoid"}:
            raise ValueError("Invalid output geometry")
    if len({o["name"] for o in c["outputs"]}) != len(c["outputs"]):
        raise ValueError("Duplicate output policy name")
    if c["phase"] == "pilot_tune" and c["domain"] == "mnist" and c["model_seed"] != 42:
        raise ValueError("MNIST tuning is restricted to model 42")
    return c


def load(path):
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    raw = json.loads(p.read_text())
    if raw.get("version") != 2 or set(raw) - {"version", "description", "defaults", "grid", "variants", "contrasts"}:
        raise ValueError("A v2 study config is required")
    base = merge(DEFAULT, raw.get("defaults", {}))
    jobs = []
    for v in raw.get("variants", []):
        if set(v) - {"label", "config", "grid"} or not v.get("label"):
            raise ValueError("Invalid variant")
        label = v["label"]
        if any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for ch in label):
            raise ValueError("Unsafe label")
        grid = dict(raw.get("grid", {}), **v.get("grid", {}))
        if any(not isinstance(values, list) or not values for values in grid.values()):
            raise ValueError("Nonempty grid lists required")
        keys = sorted(grid)
        for values in itertools.product(*(grid[k] for k in keys)):
            c = merge(base, v.get("config", {}))
            for key, value in zip(keys, values):
                assign(c, key, value)
            if c["checkpoint"]:
                prefix = "mnist" if c["model_seed"] == 42 else f"mnist_seed{c['model_seed']}"
                c["checkpoint"] = c["checkpoint"].format(mnist_run=prefix, dim=c["problem"]["dim"],
                    components=c["problem"]["components"], distribution_seed=c["distribution_seed"],
                    model_seed=c["model_seed"])
            validate(c)
            jobs.append(dict(label=label, config=c, id=f"{label}-{fingerprint(c)[:20]}"))
    if not jobs or len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Empty study or duplicate jobs")
    labels = {j["label"] for j in jobs}
    contrasts = raw.get("contrasts", [])
    if not contrasts or len({c["id"] for c in contrasts}) != len(contrasts):
        raise ValueError("Predeclared nonempty unique contrasts required")
    for contrast in contrasts:
        if not {contrast["left"], contrast["right"]} <= labels or not contrast["metrics"]:
            raise ValueError("Contrast names unknown methods or no metrics")
    return jobs, contrasts, fingerprint(raw)


def manifest(path):
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    raw = json.loads(p.read_text())
    if raw.get("format") != "dms_v2_manifest" or not raw.get("studies"):
        raise ValueError("V2 study manifest required")
    studies = [(ROOT / s).resolve() for s in raw["studies"]]
    if any(not p.is_relative_to(ROOT) or "results" in p.relative_to(ROOT).parts for p in studies):
        raise ValueError("Study configs must be inside v2 source")
    return raw, studies


def treatment(c):
    # A label with different steps/radii can NEVER become one averaged method.
    return fingerprint({k: c[k] for k in ["domain", "field", "initialization", "method", "model", "mnist"]})


def required_inputs(jobs, overrides=None):
    paths = {}
    for job in jobs:
        c = job["config"]
        if c["checkpoint"]:
            p = input_path(c["checkpoint"], overrides)
            paths[str(p)] = "trained diffusion checkpoint"
        if c["domain"] == "mnist":
            paths[str(input_path(c["mnist"]["evaluator_checkpoint"], overrides))] = "frozen independent evaluator"
            data = input_path(c["mnist"]["data_root"], overrides) / "MNIST" / "raw"
            # torchvision accepts these uncompressed raw IDX files without download.
            for name in ["train-images-idx3-ubyte", "train-labels-idx1-ubyte"]:
                paths[str(data / name)] = "local MNIST raw IDX (no download)"
    return paths
