"""Read archived arrays/metadata only; never import a trained model."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from .distributions import GMM, reference_modes
from .metrics import evaluate_modes
from .geometry import select, cluster
from .storage import file_hash

LEGACY_ARRAYS = {
    "ddim_same_particles": "ddim_same_particles_raw",
    "ddim_same_particles_merge_v3f2": "ddim_same_particles_merge_v3f2",
    "ddim_same_particles_merge_baseline": "ddim_same_particles_merge_baseline",
    "v3f2_learned_modes": "v3f2_learned",
    "v3f2_exact_modes": "v3f2_exact",
    "ddim_equal_nfe_samples": "ddim_equal_nfe_raw",
    "ddim_equal_nfe_merge_v3f2": "ddim_equal_nfe_merge_v3f2",
    "ddim_equal_nfe_merge_baseline": "ddim_equal_nfe_merge_baseline",
}


def costs_for_legacy(method, record):
    c = record.get("search_comparison", {}).get("cost", {})
    if method == "v3f2_exact":
        return dict(neural_nfe=c.get("v3f2_exact_neural_nfe_for_shared_starts"),
            analytic_evaluations=c.get("v3f2_exact_analytic_score_evaluations"))
    key = ("v3f2_learned_total_neural_nfe" if method == "v3f2_learned" else
           "ddim_equal_nfe_actual_neural_nfe" if "equal_nfe" in method else "ddim_same_particles_neural_nfe")
    return dict(neural_nfe=c.get(key), analytic_evaluations=0)


def _reduce(candidates, spec, max_points):
    if len(candidates) > max_points:
        raise ValueError("Retrospective quadratic reduction exceeds max_cluster_points")
    if spec["kind"] == "cluster":
        return cluster(candidates, spec["radius"], spec.get("representative", "medoid"),
                       spec.get("metric", "l2"), spec.get("linkage", "complete"))[0]
    cfg = dict(kind=spec["kind"], k=spec.get("k"), bandwidth=spec.get("bandwidth", 1.), iterations=spec.get("iterations", 20))
    return select(candidates, cfg)


def analyze_archives(roots, cfg, manifest=None):
    """Automatic adapter for validation NPZ; explicit manifest for v1/v2/b0/b10.

    No coordinates are reconstructed from means/stds, and absent NFE stays null.
    """
    inventory, records, arrays = [], [], {}
    entries, seen = [], set()
    for root in roots:
        root = Path(root)
        if not root.exists():
            inventory.append(dict(path=str(root), status="missing_input"))
            continue
        for p in sorted(root.rglob("*.npz")):
            if p.resolve() in seen:
                continue
            seen.add(p.resolve())
            with np.load(p, allow_pickle=False) as z:
                keys = set(z.files)
            if {"mixture_mus", "mixture_sigmas", "mixture_weights"} <= keys:
                entries.append(dict(path=p, automatic=True))
            else:
                inventory.append(dict(path=str(p), status="unrecognized_array_schema", keys=sorted(keys)))
        for p in sorted(root.rglob("*.json")):
            # Preserve inspectable numeric records with exact provenance. No
            # "recomputed" label and no uncertainty from aggregate-only files.
            try:
                obj = json.loads(p.read_text())
            except (ValueError, OSError) as exc:
                inventory.append(dict(path=str(p), status="invalid_json", error=str(exc)))
                continue
            numeric_record = p.name.startswith("summary") or isinstance(obj, dict) and "search_comparison" in obj
            if numeric_record:
                inventory.append(dict(path=str(p), status="saved_metrics_only", sha256=file_hash(p),
                    explanation="No raw coordinates in this JSON; do not infer new radii or new matching from aggregates."))
                # Store whole numeric record in separate output, not as newly
                # independent run data or synthetic observations.
                arrays[f"saved_{len(arrays):05d}"] = dict(path=str(p), content=obj)
    if manifest:
        source = Path(manifest)
        raw = json.loads(source.read_text())
        for e in raw["runs"]:
            entry = dict(e, automatic=False)
            entry["path"] = (source.parent / e["path"]).resolve()
            entries.append(entry)
    for index, entry in enumerate(entries):
        p = entry["path"]
        with np.load(p, allow_pickle=False) as z:
            if entry["automatic"]:
                means, sigmas, weights = z["mixture_mus"], z["mixture_sigmas"], z["mixture_weights"]
                means = means.reshape(len(weights), -1)
                g = GMM(means, np.eye(means.shape[1])[None] * sigmas[:, None, None] ** 2, weights)
                ref, status = reference_modes(g, restarts=cfg.get("reference_restarts", 32))
                methods = {LEGACY_ARRAYS[k]: z[k].copy() for k in LEGACY_ARRAYS if k in z.files}
                companions = list(p.parent.parent.glob("runs/" + p.stem + ".json")) + list(p.parent.parent.glob("search_runs/" + p.stem + ".json"))
                record = json.loads(companions[0].read_text()) if companions else {}
                # Evaluation retains provenance of numerical modes vs component
                # centers; it does not call the latter mathematical truth.
            else:
                ref = np.asarray(z[entry["reference_key"]]).reshape(-1, int(entry["dimension"]))
                methods = {entry["method"]: np.asarray(z[entry["candidate_key"]]).reshape(-1, ref.shape[1])}
                status = dict(complete=bool(entry.get("reference_complete", False)),
                    status=entry.get("reference_status", "externally_supplied_unverified"), rare_mask=[False] * len(ref))
                record = entry
            for method, x in methods.items():
                x = np.asarray(x).reshape(-1, ref.shape[1])
                cost = costs_for_legacy(method, record) if entry["automatic"] else dict(
                    neural_nfe=entry.get("neural_nfe"), analytic_evaluations=entry.get("analytic_evaluations"))
                for label, cand in [(method, x)] + [(method + "/" + r["label"], _reduce(x, r, cfg["max_cluster_points"])) for r in cfg["reductions"]]:
                    met = evaluate_modes(ref, cand, cfg["radii"], status.get("rare_mask"))
                    for values in met:
                        records.append(dict(source=str(p), source_sha256=file_hash(p), method=label,
                            seed=record.get("init_seed", entry.get("seed")),
                            distribution_seed=record.get("dist_seed", entry.get("distribution_seed")),
                            reference_status=status, dimension=ref.shape[1], cost=cost, **values))
    available = sorted({r["method"].split("/")[0] for r in records})
    return dict(inventory=inventory, recomputed=records, available_methods=available,
        missing_requested_methods=[m for m in ["v1", "v2", "v3f1", "v3f2", "v3f3", "b0", "b10"]
            if not any(m in a for a in available)], saved_records=arrays,
        limitation="Saved aggregate metrics are not raw samples or independent confirmation.")
