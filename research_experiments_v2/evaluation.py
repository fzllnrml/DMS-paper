"""Post-search truth/trajectory diagnostics. Never imported by the search engine."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from research_experiments.common.metrics import evaluate_modes
from research_experiments.common.distributions import reference_modes
from .problems import Problem, numerical_basin_endpoints, certify_isotropic_pair
from .cost import CountedScore
from .storage import array_hash, fingerprint


def score_identity(problem):
    return problem.identity if isinstance(problem, Problem) else fingerprint(problem.as_dict())


def evaluate_synthetic(problem, result, outputs, radii, basin_diagnostics=True, reference=None):
    # No reference has been created before run() returns.
    modes, info = reference if reference is not None else (problem.reference() if isinstance(problem, Problem)
                                                          else certify_isotropic_pair(problem))
    rows = {}
    for name, out in outputs.items():
        values = evaluate_modes(modes, out["candidates"], radii, info.get("rare_mask"))
        for row in values:
            row["reference_complete"] = info["complete"]
            if not info["complete"]:
                # Preserve only explicitly named subset diagnostics. Empty or
                # incomplete references never masquerade as a full denominator.
                for key in ["recall", "precision", "f1", "one_to_one_recall", "one_to_one_precision", "one_to_one_f1", "rare_recall", "soft_error"]:
                    row["reference_subset_" + key] = row[key]
                    row[key] = None
        rows[name] = values
    diagnostic = dict(reference=info, merge=[], split=[],
                      intervention_events_observed=result.get("intervention_events_observed", True))
    arrays = dict(modes=modes)
    if info["complete"]:
        def basins(x):
            return problem.basin_labels(x, info) if isinstance(problem, Problem) else np.full(len(x), -1, int)
        for event in result["merge_events"]:
            before = basins(event["before"])
            after = basins(event["after"])
            groups = event["groups"]
            known = [g for g in groups if len(g) > 1 and all(before[g] >= 0)]
            false = [g for g in known if len(set(before[g])) > 1]
            before_metrics = evaluate_modes(modes, event["before"], radii)
            after_metrics = evaluate_modes(modes, event["after"], radii)
            distances = cdist(modes, event["before"])
            nearest = distances.argmin(0)
            nearest_distance = distances.min(0)
            direct = []
            for left, right in zip(before_metrics, after_metrics):
                near = np.where(nearest_distance <= left["radius"], nearest, -1)
                cross = sum(len(set(near[g][near[g] >= 0])) > 1 for g in groups)
                direct.append(dict(radius=left["radius"], near_mode_cross_groups=cross,
                    n_matched_before=left["n_matched"], n_matched_after=right["n_matched"],
                    event_lost_matches=max(0, left["n_matched"]-right["n_matched"]),
                    recall_before=left["one_to_one_recall"], recall_after=right["one_to_one_recall"]))
            diagnostic["merge"].append(dict(stage=event["stage"], t=event["t"],
                nontrivial_groups=sum(len(g) > 1 for g in groups), groups_with_certified_basins=len(known),
                cross_clean_basin_groups=len(false),
                lost_clean_basin_labels=sorted(set(before[before >= 0]) - set(after[after >= 0])),
                clean_basin_labels_before=before, clean_basin_labels_after=after,
                direct_mode_metrics=direct,
                interpretation="clean Euclidean attraction basins; no ground truth was used by Merge"))
        for event in result["split_events"]:
            if event["completed"]:
                parent = basins(event["before"][None])[0]
                children = basins(event["children"])
                diagnostic["split"].append(dict(t=event["t"], parent=int(parent), children=children,
                    additional_clean_basins=len(set(children) - {parent}) if parent >= 0 and np.all(children >= 0) else None,
                    accepted=event["accepted"], cost=event["cost"]))
        initial_labels = basins(result["initial"])
        final_labels = basins(result["candidates"])
        diagnostic["initial_clean_basin_labels"] = initial_labels
        diagnostic["final_clean_basin_labels"] = final_labels
        diagnostic["certified_clean_basin_diagnostics"] = isinstance(problem, Problem) and problem.orthogonal_axes
    if basin_diagnostics and not isinstance(problem, Problem):
        diagnostic["numerical_basins_initial"] = numerical_basin_endpoints(problem, result["initial"][:16], modes)
        diagnostic["numerical_basins_final"] = numerical_basin_endpoints(problem, result["candidates"][:16], modes)
    return rows, diagnostic, arrays


def compatibility(learned, mixture, config, checkpoint_sha, evidence=None):
    """Distinguish verifiable tensor/schedule facts from missing training metadata."""
    expected_density = fingerprint(mixture.as_dict())
    state = learned.state
    declared = (evidence or {}).get("checkpoints", {}).get(checkpoint_sha)
    expected = dict(density_sha256=expected_density, coordinates="original_x",
                    beta_start=config["model"]["beta_start"], beta_end=config["model"]["beta_end"],
                    T=config["model"]["T"], activation=config["model"]["activation"],
                    prediction_target="epsilon", score_conversion="-epsilon/sqrt(1-alpha_bar_t)")
    matching = bool(declared) and all(declared.get(k) == v for k, v in expected.items())
    if declared and not matching:
        raise ValueError("Checkpoint compatibility evidence contradicts resolved target/settings")
    # Historical synthetic checkpoint contains dim/T/hidden_dims, but does NOT
    # store beta endpoints, activation, target GMM or coordinate normalization.
    return dict(verified_training_protocol=matching and bool(declared.get("evidence")),
                checkpoint_sha256=checkpoint_sha, expected=expected,
                checkpoint_dim=int(state["dim"]), checkpoint_T=int(state["T"]),
                schedule_alpha_sha256=array_hash(learned.schedule.alphas_cumprod),
                missing_from_legacy_checkpoint=["GMM parameters", "beta endpoints", "activation", "coordinate normalization", "prediction target"],
                evidence=declared,
                conclusion_scope="protocol-conditional diagnostic until training evidence is supplied")


def alignment_diagnostics(exact, learned, result, modes, ledger, points=16, seed=0):
    """Same-coordinate direction/norm errors, epsilon sign check and local probes."""
    scores = [CountedScore(field, ledger) for field in [exact, learned]]
    records, arrays = [], {}
    selected = [s for s in result["snapshots"] if s["stage"] == "initial" or s["stage"].startswith("after_level_") or s["stage"] == "final_state"]
    if len(modes):
        rng = np.random.default_rng(seed + 9901)
        perturbation = rng.standard_normal((min(points, len(modes)), exact.dim))
        perturbation /= np.maximum(np.linalg.norm(perturbation, axis=1, keepdims=True), 1e-12)
        for radius in [0., .05, .25]:
            selected.append(dict(stage=f"near_reference_{radius}", t=0,
                                 x=modes[:points] + radius * perturbation, lineage=None))
    with ledger.scope("diagnostic_alignment"):
        for i, stage in enumerate(selected):
            x = stage["x"][:points]
            if not len(x):
                continue
            a, b = [score(x, int(stage["t"]), diagnostic=True) for score in scores]
            an, bn = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
            valid = (an > 1e-6) & (bn > 1e-6)
            cos = np.full(len(x), np.nan)
            cos[valid] = np.sum(a[valid] * b[valid], axis=1) / (an[valid] * bn[valid])
            records.append(dict(stage=stage["stage"], t=stage["t"], n=len(x), n_nonzero_directions=int(valid.sum()),
                                direction_cosine=float(cos[valid].mean()) if valid.any() else None,
                                absolute_score_error=float(np.linalg.norm(a - b, axis=1).mean()),
                                exact_score_norm=float(an.mean()), learned_score_norm=float(bn.mean()),
                                relative_score_error=float(np.mean(np.linalg.norm(a[valid] - b[valid], axis=1) / an[valid])) if valid.any() else None))
            arrays.update({f"probe{i:03d}_x": x, f"probe{i:03d}_exact": a, f"probe{i:03d}_learned": b,
                           f"probe{i:03d}_cosine": cos})
        # Direct epsilon-net call is charged as an extra diagnostic neural query.
        import torch
        x = result["initial"][:points]
        t = 300
        ledger.charge(len(x), "neural", diagnostic=True)
        with torch.no_grad():
            tensor = torch.as_tensor(x, dtype=torch.float32, device=learned.device)
            times = torch.full((len(x),), t, device=learned.device, dtype=torch.long)
            eps = learned.model.net(tensor, times).cpu().numpy()
        actual = scores[1](x, t, diagnostic=True)
        converted = -eps / np.sqrt(1 - learned.schedule.alpha(t))
        sign_error = float(np.max(np.abs(actual - converted)))
    return dict(probes=records, direct_epsilon_forward_calls=1, epsilon_score_identity_max_absolute_error=sign_error,
                epsilon_score_identity_passed=bool(np.allclose(actual, converted, rtol=2e-5, atol=2e-5)),
                same_coordinate_units="original_x (no x/sqrt(alpha) rescaling)",
                near_reference_scope="available reference subset; completeness recorded separately",
                interpretation="direction/basin changes combine schedule, model and algorithm effects; do not attribute every gap to training"), arrays
