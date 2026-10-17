"""Budget/output coordinator around the frozen v1 movement and geometry kernels.

No reference modes, component parameters, labels or weights enter this API.
M5 is the M3 treatment when all operations coincide; it is not a second run.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from research_experiments.common.search import move
from research_experiments.common.geometry import cluster
from .cost import BudgetExhausted
from .adapters import split_kernel, original_search


def plan(method):
    levels = method["levels"]
    steps = method["steps"] if isinstance(method["steps"], list) else [method["steps"]] * len(levels)
    sizes = method["step_size"] if isinstance(method["step_size"], list) else [method["step_size"]] * len(levels)
    return [(f"level_{i}", t, n, size) for i, (t, n, size) in enumerate(zip(levels, steps, sizes))] + [
        ("final", method["final_t"], method["final_steps"], sizes[-1] * method["final_scale"])]


def output_candidates(x, lineage, policy):
    """Strict K uses only surviving final states, never deleted particles/padding."""
    x = np.asarray(x, float)
    _, unique = np.unique(x, axis=0, return_index=True)
    unique = sorted(unique.tolist())
    cloud = x[unique]
    ancestry = [lineage[i] for i in unique] if lineage is not None else None
    k = policy["k"]
    if policy["kind"] == "exact":
        n = min(k, len(cloud))
        y, groups, _ = cluster(cloud, 0, policy["representative"], policy["metric"], k=n) if n else (cloud, [], [])
        met = len(y) == k and len(np.unique(y, axis=0)) == k
    else:
        y, groups, _ = cluster(cloud, policy["radius"], policy["representative"], policy["metric"])
        if policy["kind"] == "at_most" and len(y) > k:
            y, outer, _ = cluster(y, 0, "medoid", policy["metric"], k=k)
            groups = [[p for g in group for p in groups[g]] for group in outer]
        met = policy["kind"] == "radius" or len(y) <= k
    output_lineage = ([sorted({parent for i in group for parent in ancestry[i]}) for group in groups]
                      if ancestry is not None else None)
    return dict(candidates=y, lineage=output_lineage, k_constraint_met=met,
                n_surviving_unique=len(cloud), policy=policy)


def run(score, starts, method, budget, P0):
    """All new starts are charged to the same ledger before being committed."""
    sampling = method["name"] == "sampling"
    count = budget["limit"] // starts.per_start if sampling and budget["regime"] == "matched" else P0
    if count < P0:
        raise ValueError("Budget cannot afford the common initial prefix")
    initial = starts(range(count))
    x = initial.copy()
    lineage = [[i] for i in range(len(x))]
    snapshots = [dict(stage="initial", t=0, x=x.copy(), lineage=deepcopy(lineage))]
    movement, stages, merges, splits, restart_events = [], [], [], [], []
    finished_plan, budget_interruption = True, False
    if method["implementation"].startswith("historical_"):
        try:
            result = original_search(score, initial, method)
            result.update(merge_events=[], restart_events=[], completed_plan=True,
                          stopped_by_budget=False, n_starts_generated=len(initial), lineage=None,
                          intervention_events_observed=False)
            return result
        except BudgetExhausted:
            # An incomplete legacy return value cannot be passed off as its output.
            return dict(candidates=initial, initial=initial, lineage=lineage,
                        snapshots=snapshots, movement=[], stages=[], split_events=[], merge_events=[],
                        restart_events=[], completed_plan=False, stopped_by_budget=True,
                        intervention_events_observed=False,
                        n_starts_generated=len(initial), warning="legacy cap interruption: provisional starts, excluded from rankings")

    def snapshot(stage, t):
        snapshots.append(dict(stage=stage, t=t, x=x.copy(), lineage=deepcopy(lineage)))

    def movement_block(stage, t, iterations, size):
        nonlocal x, lineage, budget_interruption
        try:
            with score.cost.scope("refill_refinement" if stage.startswith("refill") else "movement"):
                x, _, _ = move(score, x, t, iterations, size, method, initial, lineage,
                               movement, snapshots, stage)
            return True
        except BudgetExhausted as exc:
            x, lineage = exc.state
            budget_interruption = True
            return False

    if not sampling:
        for index, (stage, t, n, size) in enumerate(plan(method)):
            snapshot("before_" + stage, t)
            before_n = len(x)
            if not movement_block(stage, t, n, size):
                finished_plan = False
                snapshot("budget_stop_" + stage, t)
                break
            snapshot("after_" + stage, t)
            s = method["split"]
            if s["kind"] != "none" and stage != "final" and t in method["split_levels"]:
                # Commit per parent, retaining all already completed interventions.
                original_n = len(x)
                new_x, new_lineage = [], []
                interrupted = False
                for parent in range(original_n):
                    before_cost = score.cost.search_cost
                    try:
                        children = split_kernel(score, x[parent], t, s, dict(method, step_size=size))
                        capped = len(new_x) + len(children) + original_n - parent - 1 > s["max_particles"]
                        if capped:
                            children = [x[parent].copy()]
                        children = np.asarray(children).reshape(-1, score.dim)
                        splits.append(dict(t=t, parent=parent, before=x[parent].copy(), children=children,
                            accepted=len(children) > 1, capped=capped, completed=True,
                            cost=score.cost.search_cost - before_cost,
                            lineage=lineage[parent], population_before=original_n))
                        new_x.extend(children)
                        new_lineage.extend([lineage[parent]] * len(children))
                    except BudgetExhausted:
                        splits.append(dict(t=t, parent=parent, before=x[parent].copy(), children=None,
                            accepted=False, completed=False, cost=score.cost.search_cost - before_cost))
                        new_x.extend(x[parent:])
                        new_lineage.extend(lineage[parent:])
                        interrupted, budget_interruption, finished_plan = True, True, False
                        break
                x, lineage = np.asarray(new_x).reshape(-1, score.dim), deepcopy(new_lineage)
                snapshot("after_split_" + stage, t)
                if interrupted:
                    break
            g = method["merge"]
            enabled = (g["when"] in {"every", "stable"} and (index + 1) % g["every"] == 0 or
                       g["when"] == "selected" and t in g["selected"] or
                       g["when"] == "final" and stage == "final")
            # Stable means an entire same-population displacement window AND a
            # paid current-score norm check. Projection alone is no certificate.
            gate = None
            if enabled and g["when"] == "stable":
                window = [r for r in movement if r["stage"] == stage][-method["stable_window"]:]
                enabled = (len(window) == method["stable_window"] and all(r["n"] == len(x) for r in window)
                           and max(float(np.max(r["actual_l2"])) for r in window) <= g["stable_threshold"])
                if enabled:
                    try:
                        with score.cost.scope("merge_gate"):
                            gate = score(x, t)
                        enabled = bool(np.max(np.linalg.norm(gate, axis=1)) <= method["stable_score_max"])
                    except BudgetExhausted:
                        enabled, budget_interruption = False, True
            if enabled:
                radius = max(g["radius"], (g["radius_factor"] or 0) * score.schedule.sigma(t))
                directions = gate
                try:
                    if g["direction_cosine"] is not None and directions is None:
                        with score.cost.scope("merge_gate"):
                            directions = score(x, t)
                    with score.cost.scope("clustering"):
                        y, groups, reps = cluster(x, radius, g["representative"], g["metric"],
                                                 g["linkage"], directions=directions,
                                                 cosine_threshold=g["direction_cosine"])
                    merges.append(dict(stage=stage, t=t, radius=radius, metric=g["metric"],
                                       before=x.copy(), after=y.copy(), groups=groups, representatives=reps,
                                       lineage_before=deepcopy(lineage)))
                    lineage = [sorted({i for j in ids for i in lineage[j]}) for ids in groups]
                    x = y
                except BudgetExhausted:
                    budget_interruption = True
            stages.append(dict(stage=stage, t=t, before=before_n, after=len(x), cost=score.cost.search_cost))
            snapshot(stage, t)

        if budget["regime"] == "matched" and method["refill"] == "restarts":
            macro = plan(method)
            per_restart = starts.per_start + sum(n for _, _, n, _ in macro)
            # A whole new start + the same ascent schedule is an atomic policy
            # choice. Never append a dangling anchor before successful refinement.
            while score.cost.remaining >= per_restart:
                logical_id = len(initial)
                before_restart = score.cost.search_cost
                y = starts([logical_id])
                candidate_initial = np.vstack([initial, y])
                ids = [[logical_id]]
                for stage, t, n, size in macro:
                    with score.cost.scope("restart_movement"):
                        y, _, _ = move(score, y, t, n, size, method, candidate_initial, ids,
                                       movement, snapshots, f"restart_{logical_id}_{stage}")
                initial = candidate_initial
                x, lineage = np.vstack([x, y]), lineage + ids
                restart_events.append(dict(logical_start=logical_id, completed=True, cost=score.cost.search_cost-before_restart))
                snapshot(f"restart_{logical_id}", 0)
        if budget["regime"] == "matched" and method["refill"] in {"refine", "restarts"}:
            size = plan(method)[-1][-1]
            # Complete population steps, then a deterministic rotating prefix for
            # the remaining <N queries. Actual spending is exactly B if nonempty.
            full_steps, tail = divmod(score.cost.remaining, len(x)) if len(x) else (0, 0)
            if full_steps:
                movement_block("refill_final", 0, full_steps, size)
            if tail:
                offset = len(movement) % len(x)
                ids = (np.arange(tail) + offset) % len(x)
                with score.cost.scope("refill_refinement"):
                    y, _, _ = move(score, x[ids], 0, 1, size, method, initial,
                                   [lineage[i] for i in ids], movement, snapshots, "refill_partial_batch")
                x[ids] = y
            snapshot("refill_output", 0)
    snapshot("final_state", 0)
    return dict(candidates=x, initial=initial, lineage=lineage, snapshots=snapshots, movement=movement,
                stages=stages, split_events=splits, merge_events=merges, restart_events=restart_events,
                completed_plan=finished_plan, stopped_by_budget=budget_interruption,
                intervention_events_observed=True,
                n_starts_generated=len(initial),
                partial_budget_policy="retain completed moves/interventions; matched refill spends remaining queries")
