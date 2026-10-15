"""Lazy, read-only adapters to the actual legacy implementations."""
from __future__ import annotations

from copy import deepcopy
import importlib
import inspect
import numpy as np

CLASSES = {
    "v1": ("mode_finder", "ModeFinder"),
    "v2": ("mode_finder_v2", "ModeFinderV2"),
    "v3f1": ("mode_finder_v3_f1", "ModeFinderV3F1"),
    "v3f2": ("mode_finder_v3_f2", "ModeFinderV3F2"),
    "v3f3": ("mode_finder_v3_f3", "ModeFinderV3F3"),
}


class LegacyProxy:
    """Legacy reset_nfe cannot erase the independent cumulative cost ledger."""
    def __init__(self, score):
        self.counted, self.dim, self.schedule = score, score.dim, score.schedule
        self.device, self.offset, self.visits = "cpu", score.cost.search_cost, []

    @property
    def nfe(self):
        return self.counted.cost.search_cost - self.offset

    def enable_nfe_counting(self):
        self.reset_nfe()

    def reset_nfe(self):
        self.offset = self.counted.cost.search_cost

    def score_numpy(self, x, t):
        a = np.asarray(x, float).reshape(-1, self.dim)
        y = self.counted(a, t)
        self.visits.append(dict(stage="legacy_score_input", t=int(t), x=a.copy(), lineage=None))
        return y.reshape(np.asarray(x).shape)

    def score(self, x, t):
        import torch
        a = x.detach().cpu().numpy()
        y = self.score_numpy(a, int(t))
        return torch.as_tensor(y, dtype=x.dtype, device=x.device)


def get_class(kind):
    module, name = CLASSES[kind]
    return getattr(importlib.import_module("src.algorithms." + module), name)


def split_kernel(score, point, t, split_cfg, method):
    # Use the original finite-difference Hessian, eigendirection selection and
    # acceptance criterion. These methods use only score_numpy/schedule.
    kind = split_cfg["kind"]
    cls = get_class("v2" if kind == "axes" else kind)
    proxy = LegacyProxy(score)
    size = method["step_size"]
    box = (method["box"] if method["projection"] == "clamp" else None) or [-np.inf, np.inf]
    kwargs = dict(timesteps=[t], step_size=size,
        split_eps=split_cfg["delta"], max_split_directions=split_cfg["max_directions"],
        hessian_fd_eps=split_cfg["fd_eps"], ascent_steps=2 * split_cfg["probe_steps"],
        normalize_score=method["step_mode"] != "raw", x_min=box[0], x_max=box[1])
    if method["step_mode"] == "rms":
        kwargs["step_size"] *= np.sqrt(score.dim)
    if method["coordinate_mode"] != "indexed":
        raise ValueError("Legacy Split kernel requires indexed coordinates")
    if kind in {"v2", "axes"}:
        kwargs.update(split_method="axes" if kind == "axes" else "hessian",
            split_threshold=split_cfg["absolute_threshold"],
            hessian_split_eigenvalue_threshold=split_cfg["eigen_threshold"])
    else:
        kwargs.update(amplification_threshold=split_cfg["amplification"], tau_abs_min=split_cfg["absolute_threshold"])
    obj = cls(proxy, **kwargs)
    before = score.cost.search_cost
    result = obj._check_split(point.copy(), t) if kind == "axes" else obj._check_split_hessian(point.copy(), t)
    if kind != "axes":
        score.cost.event("finite_difference_score_evaluations", 2 * score.dim)
        score.cost.event("hessian_matrices")
    score.cost.event("split_total_score_evaluations", score.cost.search_cost - before)
    return result


def run_legacy(score, initial, cfg):
    kind = cfg["name"].removeprefix("historical_")
    cls, proxy = get_class(kind), LegacyProxy(score)
    box = (cfg["box"] if cfg["projection"] == "clamp" else None) or [-np.inf, np.inf]
    general = dict(timesteps=cfg["levels"], step_size=cfg["step_size"], ascent_steps=cfg["steps"],
        refine_steps=cfg["final_steps"], refine_step_scale=cfg["final_scale"],
        n_starts=len(initial), n_particles=len(initial), normalize_score=cfg["step_mode"] != "raw",
        merge_radius=cfg["merge"]["radius"], merge_radius_min=cfg["merge"]["radius"],
        merge_factor=cfg["merge"]["radius_factor"] if cfg["merge"]["radius_factor"] is not None else 0.,
        split_eps=cfg["split"]["delta"], split_threshold=cfg["split"]["absolute_threshold"],
        amplification_threshold=cfg["split"]["amplification"], tau_abs_min=cfg["split"]["absolute_threshold"],
        max_split_directions=cfg["split"]["max_directions"], hessian_fd_eps=cfg["split"]["fd_eps"],
        hessian_split_eigenvalue_threshold=cfg["split"]["eigen_threshold"],
        max_active_per_start=cfg["split"]["max_particles"], x_min=box[0], x_max=box[1])
    signature = inspect.signature(cls.__init__).parameters
    kwargs = {k: v for k, v in general.items() if k in signature}
    if cfg["step_mode"] == "rms" and "step_size" in kwargs:
        kwargs["step_size"] *= np.sqrt(score.dim)
    extra = deepcopy(cfg["historical_kwargs"])
    if set(extra) - set(signature) or set(extra) & {"self", "model", "_ignored"}:
        raise ValueError(f"Unknown legacy arguments {set(extra) - set(signature)}")
    kwargs.update(extra)
    if kind == "v3f3":
        # No reinterpretation as coarse-to-fine ascent. Preserve ODE machinery.
        kwargs.setdefault("t_start", max(cfg["levels"]))
        kwargs.setdefault("t_end", cfg["final_t"])
        kwargs.setdefault("refine_alpha", cfg["step_size"] * cfg["final_scale"])
    obj = cls(proxy, **kwargs)
    # Instrument costly legacy Hessian/trace and clustering operations on this
    # instance only, without touching source files or changing return values.
    for name, timer in [("_approximate_hessian", "hessian"), ("_compute_trace_hessian", "trace"),
                        ("_cluster", "clustering"), ("_cluster_particles", "clustering")]:
        if hasattr(obj, name):
            original = getattr(obj, name)
            def timed(*args, _fn=original, _timer=timer, **kw):
                before = score.cost.search_cost
                with score.cost.time(_timer):
                    result = _fn(*args, **kw)
                if _timer in {"hessian", "trace"}:
                    score.cost.event(_timer + "_score_evaluations", score.cost.search_cost - before)
                return result
            setattr(obj, name, timed)
    result = obj.find_modes(starts=np.asarray(initial).copy(), verbose=False)
    x = np.asarray(result.modes).reshape(-1, score.dim)
    return dict(candidates=x, initial=initial, movement=[], stages=[], split_events=[],
        snapshots=[dict(stage="initial", t=0, x=initial.copy(), lineage=None)] + proxy.visits +
            [dict(stage="output", t=cfg["final_t"], x=x.copy(), lineage=None)],
        budget_exhausted=False, historical_parameters=kwargs,
        warning="External starts replace historical built-in initialization; charge starts separately. Original class stopping rules and constructor arguments are operative.")
