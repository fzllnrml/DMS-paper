"""Post-search diagnostics. They cannot change candidates or hyperparameters."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from .search import cosine_rows


def trajectory_diagnostics(result, exact, learned=None, references=None, basins=False):
    rows = []
    previous_basins, previous_lineage = None, None
    for index, snap in enumerate(result["snapshots"]):
        x, t = snap["x"], snap["t"]
        if not len(x):
            continue
        actual_scale = np.sqrt(exact.schedule.alpha(t)) if snap.get("coordinate_mode") == "scaled" else 1.
        # Stored trajectories are the experiment's fixed coordinates. The
        # runner inserts coordinate_mode on every snapshot before diagnostics.
        sx = exact(actual_scale * x, t, diagnostic=True) * actual_scale
        row = dict(index=index, stage=snap["stage"], t=t, n=len(x),
                   exact_norm=np.linalg.norm(sx, axis=1))
        if learned is not None:
            sy = learned(actual_scale * x, t, diagnostic=True) * actual_scale
            row.update(learned_norm=np.linalg.norm(sy, axis=1), cosine=cosine_rows(sx, sy),
                relative_norm_error=np.abs(np.linalg.norm(sy, axis=1) - np.linalg.norm(sx, axis=1)) /
                    np.maximum(np.linalg.norm(sx, axis=1), 1e-12),
                zero_exact_norm=np.linalg.norm(sx, axis=1) < 1e-10)
        if references is not None and len(references):
            row["distance_to_reference"] = cdist(x, references).min(1)
        if basins:
            # Approximate basin under clean exact log-density ascent, explicitly
            # separate from labels of nearest mode and from indexed learned map.
            mixture = exact.field.mixture
            ids, residuals = [], []
            for p in x:
                opt = minimize(lambda z: -float(mixture.logpdf(z)[0]), p,
                    jac=lambda z: -mixture.score(z)[0], method="BFGS", options=dict(gtol=1e-7, maxiter=200))
                residual = float(np.linalg.norm(mixture.score(opt.x)))
                residuals.append(residual)
                if references is not None and len(references):
                    dist = np.linalg.norm(references - opt.x, axis=1)
                    ids.append(int(dist.argmin()) if residual < 1e-5 and dist.min() < 1e-3 else -1)
                else:
                    ids.append(-1)
            row.update(exact_clean_basin=ids, basin_residual=residuals,
                basin_note="Numerical clean-density optimizer; -1 means unverified. Not nearest-mode labels.")
            if previous_basins is not None and len(ids) == len(previous_basins) and snap["lineage"] == previous_lineage:
                a, b = np.asarray(previous_basins), np.asarray(ids)
                valid = (a >= 0) & (b >= 0)
                row["basin_comparable_count"] = int(valid.sum())
                row["basin_change_fraction"] = float(np.mean(a[valid] != b[valid])) if valid.any() else None
            previous_basins, previous_lineage = ids, snap["lineage"]
        rows.append(row)
    return rows


def split_diagnostics(result, references, radii):
    out = []
    if not len(references):
        return out
    for i, event in enumerate(result["split_events"]):
        before = cdist(references, np.asarray(event["before"]).reshape(1, -1)).ravel()
        after = cdist(references, event["children"]).min(1)
        population_before = cdist(references, event["population_before"]).min(1)
        population_after = cdist(references, event["population_after"]).min(1)
        for r in radii:
            new_unique = int(np.sum((population_after <= r) & (population_before > r)))
            out.append(dict(event=i, radius=r, accepted=event["accepted"], cost=event["cost"],
                local_new_reference_hits=int(np.sum((after <= r) & (before > r))),
                new_unique_reference_hits=new_unique,
                cost_per_new_unique_reference_hit=event["cost"] / new_unique if new_unique else None,
                accepted_without_immediate_new_hits=event["accepted"] and new_unique == 0,
                # Not a causal new-mode claim: later trajectories may converge.
                note="Immediate probe endpoints only; compare final recall at matched budget"))
    return out


def memory_usage():
    try:
        import resource, sys
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return dict(process_peak_rss_bytes=int(raw if sys.platform == "darwin" else raw * 1024),
                    scope="process lifetime high-water mark, not per-job allocation")
    except (ImportError, AttributeError):
        return dict(process_peak_rss_bytes=None, scope="unsupported")
