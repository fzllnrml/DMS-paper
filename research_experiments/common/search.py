"""Independent ascent and explicit population interventions.

There are no reference modes, GMM parameters or classifier labels in this API.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np
from .geometry import step_delta, cluster, select, farthest_indices
from .cost import BudgetExhausted


def cosine_rows(a, b):
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return np.divide(np.sum(a * b, axis=1), denom, out=np.zeros(len(a)), where=denom > 1e-12)


def move(score, x, t, count, size, cfg, initial, lineage, records, snapshots, stage, previous=None):
    last_shift = np.full(len(x), np.inf)
    last_score = None
    for i in range(count):
        # In scaled coordinates y=x_t/sqrt(alpha_bar_t), pass x_t=a*y to
        # the model and use grad_y log q_t(a*y)=a*score_x. Box acts on y.
        scale = np.sqrt(score.schedule.alpha(t)) if cfg["coordinate_mode"] == "scaled" else 1.
        try:
            s = scale * score(scale * x, t)
        except BudgetExhausted as exc:
            exc.state = (x.copy(), deepcopy(lineage))
            raise
        delta = step_delta(s, size, cfg["step_mode"])
        proposal = x + delta
        clipped = proposal if cfg["box"] is None else np.clip(proposal, *cfg["box"])
        updated = clipped if cfg["projection"] == "clamp" else proposal
        actual = updated - x
        last_shift = np.linalg.norm(actual, axis=1)
        anchors = np.array([initial[ids].mean(0) for ids in lineage])
        records.append(dict(stage=stage, t=t, step=i, n=len(x),
            score_norm=np.linalg.norm(s, axis=1), proposed_l2=np.linalg.norm(delta, axis=1),
            actual_l2=last_shift, actual_rms=last_shift / np.sqrt(x.shape[1]),
            displacement=np.linalg.norm(updated - anchors, axis=1),
            direction_cosine=cosine_rows(s, previous) if previous is not None else None,
            reversal_fraction=float(np.mean(cosine_rows(s, previous) < 0)) if previous is not None else None,
            outside_fraction=float(np.mean(proposal != clipped)),
            fraction_clipped=float(np.mean(proposal != updated)),
            overshoot_max=float(np.abs(proposal - clipped).max()) if len(x) else 0.,
            cost=score.cost.search_cost, coordinate_mode=cfg["coordinate_mode"]))
        x, previous, last_score = updated, s.copy(), s.copy()
        if cfg["save_every_step"]:
            snapshots.append(dict(stage=stage, t=t, step=i, x=x.copy(), lineage=deepcopy(lineage)))
        if len(x) == 0 or cfg["tolerance"] and last_shift.max() < cfg["tolerance"]:
            break
    return x, last_shift, last_score


def split_points(score, x, t, cfg, method):
    """Reuse the original v2/v3f1 split kernel; expose cost and child coordinates.

    The surrounding batch population experiment is explicitly a NEW intervention,
    distinct from historical per-start v2 (available via historical adapters).
    """
    from .historical import split_kernel
    output, parents, events = [], [], []
    for parent, point in enumerate(x):
        if cfg["boundary_only"]:
            norm = float(np.linalg.norm(score(point[None], t)))
            if norm > cfg["boundary_norm"]:
                output.append(point)
                parents.append(parent)
                continue
        before = score.cost.search_cost
        with score.cost.time("split"):
            children = split_kernel(score, point, t, cfg, method)
        if len(output) + len(children) + (len(x) - parent - 1) > cfg["max_particles"]:
            children = [point.copy()]
            capped = True
        else:
            capped = False
        before_population = np.vstack([np.asarray(output).reshape(-1, score.dim), x[parent:]])
        output.extend(children)
        parents.extend([parent] * len(children))
        events.append(dict(t=t, parent=parent, before=point.copy(), children=np.asarray(children),
            accepted=len(children) > 1, capped=capped, cost=score.cost.search_cost - before,
            population_before=before_population,
            population_after=np.vstack([np.asarray(output).reshape(-1, score.dim), x[parent + 1:]])))
        score.cost.event("split_checks")
        score.cost.event("split_children", len(children) if len(children) > 1 else 0)
        score.cost.event("split_accepted", int(len(children) > 1))
    return np.asarray(output), parents, events


def run_search(score, initial, cfg, replenish=None, seed=0):
    x = np.asarray(initial, float).copy()
    lineage = [[i] for i in range(len(x))]
    records, stages, events = [], [], []
    snapshots = [dict(stage="initial", t=0, x=x.copy(), lineage=deepcopy(lineage))]
    sizes = cfg["step_size"] if isinstance(cfg["step_size"], list) else [cfg["step_size"]] * len(cfg["levels"])
    steps = cfg["steps"] if isinstance(cfg["steps"], list) else [cfg["steps"]] * len(cfg["levels"])
    plan = [(f"level_{j}", t, n, a) for j, (t, n, a) in enumerate(zip(cfg["levels"], steps, sizes))]
    plan += [("final", cfg["final_t"], cfg["final_steps"], sizes[-1] * cfg["final_scale"])]
    if cfg["name"] == "sampling":
        plan = [("final", cfg["final_t"], 0, 0.)]
    exhausted, previous_score = False, None
    try:
        for index, (stage, t, n, size) in enumerate(plan):
            before = x.copy()
            snapshots.append(dict(stage="before_" + stage, t=t, x=x.copy(), lineage=deepcopy(lineage)))
            x, shifts, directions = move(score, x, t, n, size, cfg, initial, lineage, records, snapshots, stage, previous_score)
            previous_score = directions
            snapshots.append(dict(stage="after_" + stage, t=t, x=x.copy(), lineage=deepcopy(lineage)))
            s = cfg["split"]
            if s["kind"] != "none" and stage != "final" and (index + 1) % s["every"] == 0:
                child, parents, ev = split_points(score, x, t, s, dict(cfg, step_size=size))
                lineage = [lineage[p] for p in parents]
                x = child
                events.extend(ev)
                shifts, directions = np.full(len(x), np.inf), None
                previous_score = None
            g = cfg["merge"]
            enabled = ((stage == "final" and g["when"] != "none") or
                (stage != "final" and g["when"] == "every" and (index + 1) % g["every"] == 0) or
                (stage != "final" and g["when"] == "selected" and t in g["selected"]) or
                (stage != "final" and g["when"] == "stable" and np.all(shifts <= g["stable_threshold"])))
            merge_info = None
            if enabled:
                radius = max(g["radius"], g["radius_factor"] * score.schedule.sigma(t)) if g["radius_factor"] is not None else g["radius"]
                if g["direction_cosine"] is not None:
                    directions = score(x, t)
                with score.cost.time("clustering"):
                    merged, groups, reps = cluster(x, radius, g["representative"], g["metric"], g["linkage"],
                        directions=directions, cosine_threshold=g["direction_cosine"])
                merge_info = dict(radius=radius, groups=groups, representatives=reps, before=len(x), after=len(merged))
                lineage = [sorted(set(i for p in ids for i in lineage[p])) for ids in groups]
                x = merged
                previous_score = None
                score.cost.event("merge_removed", merge_info["before"] - merge_info["after"])
            stages.append(dict(stage=stage, t=t, before=len(before), after=len(x), merge=merge_info, cost=score.cost.search_cost))
            snapshots.append(dict(stage=stage, t=t, x=x.copy(), lineage=deepcopy(lineage)))
    except BudgetExhausted as exc:
        # Preserve every fully completed movement, even if a later call/split
        # cannot be afforded. A partially computed split is not applied.
        exhausted = True
        if hasattr(exc, "state"):
            x, lineage = exc.state
    if cfg["budget_policy"] == "continue" and not exhausted:
        try:
            x, _, _ = move(score, x, cfg["final_t"], cfg["extra_steps"], sizes[-1] * cfg["final_scale"],
                cfg, initial, lineage, records, snapshots, "reallocated_refinement")
        except BudgetExhausted as exc:
            exhausted = True
            if hasattr(exc, "state"):
                x, lineage = exc.state
    if cfg["budget_policy"] in {"restarts", "farthest_jitter"} and not exhausted:
        rng = np.random.default_rng(seed + 80801)
        for j in range(cfg["max_extra_starts"]):
            try:
                if cfg["budget_policy"] == "restarts":
                    if replenish is None:
                        raise ValueError("Independent restart factory missing")
                    y = replenish(j)
                else:
                    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
                    np.fill_diagonal(d, np.inf)
                    idx = int(d.min(1).argmax())
                    y = x[idx:idx+1] + rng.normal(0, sizes[-1], (1, x.shape[1]))
                initial = np.vstack([initial, y])
                ids = [[len(initial) - 1]]
                y, _, _ = move(score, y, cfg["final_t"], cfg["extra_steps"], sizes[-1] * cfg["final_scale"],
                    cfg, initial, ids, records, snapshots, f"restart_{j}")
                x, lineage = np.vstack([x, y]), lineage + ids
                score.cost.event("extra_starts")
            except BudgetExhausted:
                exhausted = True
                break
    snapshots.append(dict(stage="before_selection", t=cfg["final_t"], x=x.copy(), lineage=deepcopy(lineage)))
    with score.cost.time("selection"):
        x = select(x, cfg["selection"])
    # Selection may move points (KDE); no invented ancestry after this point.
    snapshots.append(dict(stage="output", t=cfg["final_t"], x=x.copy(), lineage=None))
    return dict(candidates=x, initial=initial, movement=records, stages=stages,
                snapshots=snapshots, split_events=events, budget_exhausted=exhausted)
