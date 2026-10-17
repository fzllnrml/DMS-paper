"""V2-only aggregation, paired contrasts and Pareto tables.

No old CSV can enter this loader. One particle is never a statistical replicate.
Treatment signatures retain every method/initialization/evaluation setting.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from research_experiments.common.metrics import pareto_mask
from .storage import RESULTS, FORMAT, output_path, completed, write_json, write_csv, fingerprint

METRICS = ["one_to_one_recall", "one_to_one_precision", "one_to_one_f1", "rare_recall",
           "soft_error", "matched_localization_error", "duplicate_fraction", "n_matched", "K", "queries",
           "search_seconds", "cross_clean_basin_groups", "merge_near_mode_cross_groups", "merge_event_lost_matches",
           "pixel_knn", "feature_knn", "pixel_support", "feature_support",
           "class_entropy", "class_coverage", "class_change_fraction", "pairwise_rms", "within_class_rms",
           "near_duplicate_fraction", "displacement_from_ddim_rms", "pixel_train_extreme_close_fraction",
           "feature_train_extreme_close_fraction", "pixel_train_validation_nearest_ratio", "feature_train_validation_nearest_ratio"]
STRATUM = ["cohort", "domain", "field", "phase", "study", "dimension", "family", "budget", "regime",
           "output_policy", "output_geometry", "radius", "P0", "zero_convention"]
PAIR = ["cohort", "domain", "phase", "study", "dimension", "family", "distribution_seed", "search_seed", "model_seed",
        "budget", "regime", "output_geometry", "radius", "P0", "zero_convention", "density_identity", "checkpoint_sha256"]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def cluster_id(row):
    return (row["model_seed"], row["checkpoint_sha256"]) if row["domain"] == "mnist" else row["distribution_seed"]


def interval(groups, repeats=5000, alpha=.05, seed=991, fixed_clusters=False):
    """Vectorized equal-cluster then equal-search bootstrap; no fake one-run CI."""
    groups = [np.asarray(g, float) for g in groups if len(g)]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g)]
    n = sum(map(len, groups))
    if not n:
        return dict(mean=None, sd=None, ci_low=None, ci_high=None, n_clusters=0, n_search_runs=0)
    means = np.asarray([g.mean() for g in groups])
    observed = float(means.mean())
    info = dict(mean=observed, sd=float(np.std(np.concatenate(groups), ddof=1)) if n > 1 else None,
                between_cluster_sd=float(means.std(ddof=1)) if len(means) > 1 else None,
                n_clusters=len(groups), n_search_runs=n, alpha=alpha, bootstrap_repeats=repeats,
                scope=("conditional on this fixed pretrained model set; search seeds resampled within each model"
                       if fixed_clusters else "distribution clusters, then search seeds" if len(groups) > 1
                       else "conditional on one distribution/model"))
    if n == 1 or fixed_clusters and all(len(g) == 1 for g in groups):
        return dict(info, ci_low=None, ci_high=None, interval_status="no search replication")
    tail_draws = repeats * alpha / 2
    if tail_draws < 10:
        return dict(info, ci_low=None, ci_high=None, interval_status="insufficient bootstrap tail resolution",
                    expected_tail_draws=tail_draws)
    rng = np.random.default_rng(seed)
    selected = (np.tile(np.arange(len(groups)), (repeats, 1)) if fixed_clusters
                else rng.integers(0, len(groups), (repeats, len(groups))))
    within = np.zeros((repeats, len(groups)))
    # Repeated occurrences of one selected cluster receive independent inner
    # resamples, rather than reusing one mean and overstating its covariance.
    for column in range(len(groups)):
        for index, g in enumerate(groups):
            rr = np.flatnonzero(selected[:, column] == index)
            within[rr, column] = g[rng.integers(0, len(g), (len(rr), len(g)))].mean(1)
    estimates = within.mean(1)
    lo, hi = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return dict(info, ci_low=float(lo), ci_high=float(hi), interval_status="approximate bootstrap",
                expected_tail_draws=tail_draws)


def load_rows(root):
    root = output_path(root)
    rows, seen = [], set()
    for marker in sorted(root.rglob("DONE.json")):
        directory = marker.parent
        metadata_file = directory / "metadata.json"
        if not metadata_file.exists():
            continue
        meta = json.loads(metadata_file.read_text())
        if meta.get("format") != FORMAT or meta.get("result_origin") != "new_v2_execution":
            raise ValueError(f"Non-v2 job in v2 analysis root: {directory}")
        if not completed(directory):
            raise ValueError(f"Invalid completion hashes: {directory}")
        for row in json.loads((directory / "metrics.json").read_text()):
            if row.get("format") != FORMAT or row["cohort"] != meta["cohort"]:
                raise ValueError("Metric/provenance identity mismatch")
            key = (row["cohort"], row["job_id"], row["output_geometry"], row["radius"])
            if key in seen:
                raise ValueError("Duplicate job/radius/output in analysis roots")
            seen.add(key)
            rows.append(row)
    return rows


def summary(rows, repeats=5000):
    buckets = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in STRATUM + ["treatment", "label"])
        buckets[key].append(r)
    out = []
    for key, runs in sorted(buckets.items(), key=lambda item: str(item[0])):
        group = dict(zip(STRATUM + ["treatment", "label"], key))
        eligible = [r for r in runs if r["ranking_eligible"]]
        base = dict(group, n_rows=len(runs), n_eligible=len(eligible),
                    n_nfe_violations=sum(not r["nfe_valid"] for r in runs),
                    n_k_violations=sum(not r["k_constraint_met"] for r in runs),
                    n_incomplete_references=sum(r["reference_complete"] is False for r in runs),
                    n_unverified_compatibility=sum(not r["compatibility_verified"] for r in runs),
                    n_training_checkpoints=len({r["checkpoint_sha256"] for r in runs if r["checkpoint_sha256"]}),
                    particles_are_replicates=False)
        for metric in METRICS:
            by_cluster = defaultdict(list)
            for r in eligible:
                if finite(r.get(metric)):
                    by_cluster[cluster_id(r)].append(r[metric])
            stats = interval(list(by_cluster.values()), repeats, fixed_clusters=group["domain"] == "mnist")
            if stats["n_search_runs"] or metric in {"K", "queries"}:
                out.append(dict(base, metric=metric, **stats))
    return out


def eligible_pair(a, b, spec):
    reasons = []
    if not a["ranking_eligible"] or not b["ranking_eligible"]:
        reasons.append("ineligible NFE/K/algorithm/reference/compatibility")
    if a["common_noise_sha256"] != b["common_noise_sha256"]:
        reasons.append("common noise prefix mismatch")
    if spec.get("pairing", "starts") == "starts" and a["common_initial_sha256"] != b["common_initial_sha256"]:
        reasons.append("initial state prefix mismatch")
    if a["regime"] == "matched" and abs(a["queries"] - b["queries"]) > max(a["nfe_tolerance"], b["nfe_tolerance"]):
        reasons.append("actual queries not matched")
    return reasons


def planned_primary_count(groups):
    """Multiplicity uses the predeclared design, including missing/failed cells."""
    count = 0
    for g in groups:
        for spec in g["contrasts"]:
            filters = spec.get("primary_filters")
            if not filters:
                continue
            cells = set()
            for j in g["jobs"]:
                c = j["config"]
                if j["label"] != spec["left"]:
                    continue
                row = dict(domain=c["domain"], phase=c["phase"], study=c["study"], dimension=c["problem"]["dim"],
                           family=c["problem"]["family"], budget=c["budget"]["limit"], regime=c["budget"]["regime"], P0=c["population"])
                for p in c["outputs"]:
                    for radius in [None] if c["domain"] == "mnist" else c["evaluation"]["radii"]:
                        candidate = dict(row, output_policy=p["name"], radius=radius)
                        if matches(candidate, filters):
                            cells.add(tuple(candidate.items()))
            count += len(cells) * len(spec.get("primary_metrics", spec["metrics"]))
    return max(1, count)


def matches(row, filters):
    return all(row.get(k) in values for k, values in filters.items())


def contrasts(rows, groups, repeats=5000, primary_repeats=100000):
    reports, differences = [], []
    multiplicity = planned_primary_count(groups)
    for g in groups:
        studies = {j["config"]["study"] for j in g["jobs"]}
        for spec in g["contrasts"]:
            sides = {}
            for label in [spec["left"], spec["right"]]:
                by_pair = {}
                for r in rows:
                    if r["label"] != label or r["study"] not in studies:
                        continue
                    key = tuple(r.get(k) for k in PAIR)
                    if key in by_pair:
                        raise ValueError(f"Ambiguous pairing for {label}; settings may not be averaged")
                    by_pair[key] = r
                sides[label] = by_pair
            left, right = sides[spec["left"]], sides[spec["right"]]
            cells = defaultdict(list)
            exclusions = defaultdict(CounterLike)
            for key in left.keys() | right.keys():
                a, b = left.get(key), right.get(key)
                exemplar = a or b
                # Field differs legitimately in exact/learned contrasts.
                keys = [k for k in STRATUM if k != "field"]
                cell = tuple(exemplar.get(k) for k in keys)
                if a is None or b is None:
                    exclusions[cell].add("missing paired method")
                    continue
                reasons = eligible_pair(a, b, spec)
                if reasons:
                    for reason in reasons:
                        exclusions[cell].add(reason)
                    for metric in spec["metrics"]:
                        if finite(a.get(metric)) and finite(b.get(metric)):
                            differences.append(dict(zip(keys, cell), contrast=spec["id"], metric=metric,
                                left_job=a["job_id"], right_job=b["job_id"], left=a[metric], right=b[metric],
                                delta=a[metric] - b[metric], ranking_eligible=False, exclusion_reasons=reasons))
                    continue
                cells[cell].append((a, b))
            if not cells and not exclusions:
                reports.append(dict(contrast=spec["id"], left=spec["left"], right=spec["right"],
                                    status="no completed paired jobs", n_pairs=0, study=sorted(studies)))
            for cell in cells.keys() | exclusions.keys():
                keys = [k for k in STRATUM if k != "field"]
                stratum = dict(zip(keys, cell))
                primary_cell = bool(spec.get("primary_filters")) and matches(stratum, spec["primary_filters"])
                pairs = cells.get(cell, [])
                for metric in spec["metrics"]:
                    primary = primary_cell and metric in spec.get("primary_metrics", spec["metrics"])
                    by_cluster = defaultdict(list)
                    missing_metric = 0
                    for a, b in pairs:
                        if finite(a.get(metric)) and finite(b.get(metric)):
                            delta = a[metric] - b[metric]
                            by_cluster[cluster_id(a)].append(delta)
                            differences.append(dict(stratum, contrast=spec["id"], metric=metric,
                                                    distribution_seed=a["distribution_seed"], model_seed=a["model_seed"],
                                                    search_seed=a["search_seed"], left_job=a["job_id"], right_job=b["job_id"],
                                                    left=a[metric], right=b[metric], delta=delta, ranking_eligible=True))
                        else:
                            missing_metric += 1
                    stats = interval(list(by_cluster.values()), primary_repeats if primary else repeats,
                                     fixed_clusters=stratum["domain"] == "mnist")
                    adjusted = (interval(list(by_cluster.values()), primary_repeats, alpha=.05 / multiplicity,
                                         fixed_clusters=stratum["domain"] == "mnist") if primary else {})
                    reports.append(dict(stratum, contrast=spec["id"], left=spec["left"], right=spec["right"],
                                        metric=metric, delta_definition="left minus right", pairing=spec.get("pairing", "starts"),
                                        status="ok" if stats["n_search_runs"] else "no eligible numeric pairs",
                                        primary=primary, inference="exploratory" if stratum["phase"] != "confirmation" else "locked confirmation",
                                        excluded=exclusions[cell].counts, missing_metric=missing_metric,
                                        n_pairs=stats["n_search_runs"],
                                        familywise_method="Bonferroni over predeclared primary design" if primary else None,
                                        familywise_m=multiplicity if primary else None,
                                        familywise_alpha=.05/multiplicity if primary else None,
                                        familywise_ci_low=adjusted.get("ci_low"), familywise_ci_high=adjusted.get("ci_high"),
                                        familywise_interval_status=adjusted.get("interval_status"), **stats))
    return reports, differences


class CounterLike:
    def __init__(self):
        self.counts = {}
    def add(self, reason):
        self.counts[reason] = self.counts.get(reason, 0) + 1


def pareto(rows):
    """Per-paired-problem fronts; never mix dimensions, K protocols or radii."""
    buckets, out = defaultdict(list), []
    for r in rows:
        if r["ranking_eligible"]:
            keys = PAIR + ["field"]  # never price analytic and neural queries as interchangeable
            buckets[tuple(r.get(k) for k in keys)].append(r)
    for values in buckets.values():
        mnist = values[0]["domain"] == "mnist"
        metrics = ["pixel_knn", "feature_knn", "K", "queries"] if mnist else ["one_to_one_recall", "soft_error", "K", "queries"]
        usable = [r for r in values if all(finite(r.get(m)) for m in metrics)]
        if not usable:
            continue
        mask = pareto_mask([[r[m] for m in metrics] for r in usable], [True] * 4 if mnist else [False, True, True, True])
        out.extend(dict(r, pareto=bool(keep), pareto_metrics=metrics) for r, keep in zip(usable, mask))
    return out


def aggregate_pareto(rows, repeats=5000):
    """Balanced common eligible units; never rank means from unequal subsets."""
    buckets, out, status = defaultdict(lambda: defaultdict(list)), [], []
    for r in rows:
        key = tuple(r.get(k) for k in STRATUM)
        buckets[key][(r["label"], r["treatment"])].append(r)
    for stratum, treatments in buckets.items():
        info = dict(zip(STRATUM, stratum))
        mnist = info["domain"] == "mnist"
        axes = ["pixel_knn", "feature_knn", "K", "queries"] if mnist else ["one_to_one_recall", "soft_error", "K", "queries"]
        valid = {}
        for label, values in treatments.items():
            valid[label] = {(r["distribution_seed"], r["search_seed"], r["model_seed"], r["checkpoint_sha256"]): r
                            for r in values if r["ranking_eligible"] and all(finite(r.get(m)) for m in axes)}
        common = set.intersection(*(set(v) for v in valid.values())) if valid else set()
        status.append(dict(info, n_completed_treatments=len(treatments), n_common_eligible_units=len(common),
                           status="balanced common eligible units" if common else "no common eligible units; no aggregate frontier"))
        candidates = []
        for (label, signature), by_unit in valid.items():
            if not common:
                continue
            values = [by_unit[k] for k in sorted(common, key=str)]
            row = dict(info, label=label, treatment=signature, n_runs=len(common),
                       n_completed=len(treatments[(label, signature)]),
                       interpretation="front of means on common eligible completed units; conditional, not a confidence statement")
            for metric in axes:
                clusters = defaultdict(list)
                for r in values:
                    clusters[cluster_id(r)].append(r[metric])
                stats = interval(list(clusters.values()), repeats, fixed_clusters=mnist)
                for stat in ["mean", "sd", "ci_low", "ci_high", "n_clusters", "n_search_runs"]:
                    row[metric + "_" + stat] = stats.get(stat)
            candidates.append(row)
        if candidates:
            mask = pareto_mask([[r[m + "_mean"] for m in axes] for r in candidates], [True]*4 if mnist else [False, True, True, True])
            out.extend(dict(row, pareto=bool(keep)) for row, keep in zip(candidates, mask))
    return out, status


def design_audit(rows, groups):
    """Report missing planned jobs instead of silently treating partial runs as complete."""
    def observed_key(r):
        return tuple(r.get(k) for k in ["label", "domain", "field", "phase", "study", "dimension", "family",
                                        "distribution_seed", "search_seed", "model_seed", "P0", "budget", "regime", "treatment"])
    by_cohort = defaultdict(list)
    for r in rows:
        by_cohort[r["cohort"]].append(r)
    reports = []
    for cohort, completed_rows in by_cohort.items():
        observed = {observed_key(r) for r in completed_rows}
        for g in groups:
            expected = set()
            studies = set()
            for j in g["jobs"]:
                c = j["config"]
                studies.add(c["study"])
                from .config import treatment
                expected.add((j["label"], c["domain"], c["field"], c["phase"], c["study"], c["problem"]["dim"],
                              c["problem"]["family"], c["distribution_seed"], c["seed"], c["model_seed"], c["population"],
                              c["budget"]["limit"], c["budget"]["regime"], treatment(c)))
            # Exact CPU and neural/MPS studies may be separate execution cohorts.
            if not any(r["study"] in studies for r in completed_rows):
                continue
            reports.append(dict(cohort=cohort, study=sorted(studies), expected_jobs=len(expected),
                                completed_matching_jobs=len(expected & observed), missing_jobs=len(expected-observed),
                                status="complete planned cohort" if expected <= observed else "partial cohort; missing jobs are not imputed"))
    if not reports:
        reports.append(dict(status="no completed jobs", expected_jobs=sum(len(g["jobs"]) for g in groups), completed_matching_jobs=0))
    return reports


def plots(aggregates, destination):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output_path(RESULTS / "cache" / "matplotlib")))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for r in aggregates:
        if (r["metric"] == "one_to_one_recall" and r["mean"] is not None and r["radius"] == .25
                and r["output_policy"] in {"native", "le8"} and r["study"] in {"exact_mechanisms", "confirmation_exact"}):
            keys = [k for k in STRATUM if k not in {"field"}]
            groups[tuple(r.get(k) for k in keys)].append(r)
    for key, values in groups.items():
        fig, ax = plt.subplots(figsize=(7, max(3, len(values) * .32)))
        for i, r in enumerate(sorted(values, key=lambda r: r["label"])):
            lo, hi, mean = r.get("ci_low"), r.get("ci_high"), r["mean"]
            error = None if lo is None else [[max(0., mean - lo)], [max(0., hi - mean)]]
            ax.errorbar(mean, i, xerr=error, fmt="o", label=None)
        ax.set_yticks(range(len(values)), [r["label"] for r in sorted(values, key=lambda r: r["label"])])
        ax.set(xlabel="One-to-one recall (complete reference only)", xlim=(-.03, 1.03),
               title=f"{values[0]['dimension']}D {values[0]['family']} B={values[0]['budget']} {values[0]['output_policy']}")
        fig.tight_layout()
        p = output_path(Path(destination) / (fingerprint(key)[:20] + ".png"))
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=140)
        plt.close(fig)


def pareto_plots(fronts, destination):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output_path(RESULTS / "cache" / "matplotlib")))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for r in fronts:
        if r["output_policy"] != "native" or r["radius"] not in {.25, None}:
            continue
        keys = [k for k in STRATUM if k != "budget"]
        groups[tuple(r.get(k) for k in keys)].append(r)
    for key, values in groups.items():
        mnist = values[0]["domain"] == "mnist"
        xkey, ykey = ("pixel_knn_mean", "feature_knn_mean") if mnist else ("queries_mean", "one_to_one_recall_mean")
        fig, ax = plt.subplots(figsize=(8, 5))
        scatter = ax.scatter([r[xkey] for r in values], [r[ykey] for r in values],
                             c=[r["K_mean"] for r in values], cmap="viridis", s=45)
        # Tables give the exact labels for every point. Annotate a bounded number
        # of nondominated points so a large tuning grid stays legible.
        for r in [r for r in values if r["pareto"]][:10]:
            ax.annotate(r["label"], (r[xkey], r[ykey]), xytext=(4, 5), textcoords="offset points", fontsize=7)
        fig.colorbar(scatter, ax=ax, label="Mean output candidates K")
        ax.set(xlabel="Pixel kNN (RMS) ↓" if mnist else "Intrinsic analytic score queries",
               ylabel="Feature kNN (L2) ↓" if mnist else "One-to-one recall ↑",
               title=f"{values[0]['study']}: {values[0]['dimension']}D {values[0]['family']} ({values[0]['regime']})")
        if not mnist:
            ax.set_ylim(-.03, 1.03)
        fig.tight_layout()
        p = output_path(Path(destination) / (fingerprint(key)[:20] + ".png"))
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)


def mnist_guards(rows):
    """Predeclared descriptive guard checks on the fixed model/search pairs."""
    from .protocol import GUARDS
    selected = [r for r in rows if r["domain"] == "mnist" and r["output_policy"] == "native" and r["ranking_eligible"]]
    buckets, reports = defaultdict(list), []
    for r in selected:
        buckets[(r["cohort"], r["study"], r["phase"], r["regime"])].append(r)
    for context, runs in buckets.items():
        baseline = {(r["model_seed"], r["search_seed"]): r for r in runs if r["label"] == "DDIM105"}
        for label in sorted({r["label"] for r in runs if not r["label"].startswith("DDIM")}):
            pairs = [(r, baseline[(r["model_seed"], r["search_seed"])]) for r in runs
                     if r["label"] == label and (r["model_seed"], r["search_seed"]) in baseline]
            if not pairs:
                continue
            tests = {
                "entropy": ("class_entropy", lambda a, b: a >= b-GUARDS["max_entropy_drop"]),
                "diversity": ("pairwise_rms", lambda a, b: a >= b*GUARDS["min_pairwise_rms_ratio"]),
                "near_duplicates": ("near_duplicate_fraction", lambda a, b: a <= b+GUARDS["max_near_duplicate_increase"]),
                "pixel_train_close": ("pixel_train_extreme_close_fraction", lambda a, b: a <= b+GUARDS["max_train_extreme_close_increase"]),
                "feature_train_close": ("feature_train_extreme_close_fraction", lambda a, b: a <= b+GUARDS["max_train_extreme_close_increase"])}
            failures, missing = [], []
            for name, (metric, predicate) in tests.items():
                if not all(finite(a.get(metric)) and finite(b.get(metric)) for a, b in pairs):
                    missing.append(name)
                elif not predicate(np.mean([a[metric] for a, _ in pairs]), np.mean([b[metric] for _, b in pairs])):
                    failures.append(name)
            reports.append(dict(zip(["cohort", "study", "phase", "regime"], context), label=label,
                                n_pairs=len(pairs), failed_guards=failures, missing_metrics=missing,
                                guards_passed=not failures and not missing,
                                scope="predeclared descriptive guard, not an independent confidence test or memorization proof"))
    return reports


def hypothesis_decisions(paired, hypotheses, audit, guards):
    decisions = []
    for h in hypotheses or []:
        for i, clause in enumerate(h["clauses"]):
            study_audit = [a for a in audit if clause["study"] in a.get("study", [])]
            complete_design = bool(study_audit) and all(a.get("missing_jobs") == 0 for a in study_audit)
            candidates = [r for r in paired if r.get("study") == clause["study"] and r.get("contrast") == clause["contrast"]
                          and r.get("metric") == clause["metric"] and r.get("primary")
                          and matches(r, clause.get("filters", {}))]
            if not candidates:
                decisions.append(dict(hypothesis=h["id"], clause=i, status="no eligible locked primary cells", complete_design=complete_design))
            for r in candidates:
                lo, hi = r.get("familywise_ci_low"), r.get("familywise_ci_high")
                if not complete_design:
                    status = "incomplete planned confirmation"
                elif lo is None or hi is None:
                    status = "insufficient interval information"
                elif clause["direction"] == "greater":
                    status = "supports threshold" if lo > clause["threshold"] else "below threshold" if hi <= clause["threshold"] else "inconclusive"
                else:
                    status = "supports threshold" if hi < clause["threshold"] else "above threshold" if lo >= clause["threshold"] else "inconclusive"
                guard_labels = h.get("required_guard_labels", [])
                relevant_guards = [g for g in guards if g["study"] == clause["study"] and g["label"] in guard_labels]
                guard_valid = not guard_labels or {g["label"] for g in relevant_guards} == set(guard_labels) and all(g["guards_passed"] for g in relevant_guards)
                decisions.append(dict(hypothesis=h["id"], clause=i, contrast=clause["contrast"], metric=clause["metric"],
                                      dimension=r["dimension"], family=r["family"], mean_delta=r["mean"],
                                      familywise_ci_low=lo, familywise_ci_high=hi, direction=clause["direction"], threshold=clause["threshold"],
                                      complete_design=complete_design, required_guards_passed=guard_valid,
                                      status=status if guard_valid else "guard failed or missing; support-only gain cannot establish the hypothesis"))
    return decisions


def analyze(root, destination, groups, repeats=5000, primary_repeats=100000, make_plots=False, hypotheses=None):
    rows = load_rows(root)
    aggregates = summary(rows, repeats)
    paired, deltas = contrasts(rows, groups, repeats, primary_repeats)
    destination = output_path(destination)
    write_csv(destination / "runs.csv", rows, None if rows else ["format", "job_id", "status"])
    write_csv(destination / "aggregates.csv", aggregates, None if aggregates else ["metric", "mean", "n_search_runs"])
    write_csv(destination / "paired_contrasts.csv", paired)
    write_csv(destination / "paired_differences.csv", deltas, None if deltas else ["contrast", "metric", "delta"])
    fronts = pareto(rows)
    write_csv(destination / "pareto.csv", fronts, None if fronts else ["job_id", "pareto"])
    pooled_fronts, frontier_status = aggregate_pareto(rows, repeats)
    write_csv(destination / "pareto_status.csv", frontier_status, None if frontier_status else ["status", "n_common_eligible_units"])
    audit = design_audit(rows, groups)
    write_csv(destination / "design_audit.csv", audit)
    guards = mnist_guards(rows)
    write_csv(destination / "mnist_guards.csv", guards, None if guards else ["label", "guards_passed", "scope"])
    decisions = hypothesis_decisions(paired, hypotheses, audit, guards)
    write_csv(destination / "hypothesis_decisions.csv", decisions, None if decisions else ["hypothesis", "status"])
    write_csv(destination / "pareto_aggregates.csv", pooled_fronts, None if pooled_fronts else ["label", "pareto"])
    report = dict(format="dms_v2_analysis", root=str(output_path(root)), n_jobs=len({(r["cohort"], r["job_id"]) for r in rows}),
                  n_rows=len(rows), n_eligible=sum(r["ranking_eligible"] for r in rows),
                  n_contrast_rows=len(paired), n_actual_pairs=len(deltas),
                  declared_primary_m=planned_primary_count(groups), bootstrap_repeats=repeats,
                  primary_bootstrap_repeats=primary_repeats,
                  status="no completed results; contrasts are declared, not estimated" if not rows else "analyzed",
                  cohorts=sorted({r["cohort"] for r in rows}),
                  independence="distribution seeds; MNIST intervals conditional on fixed models; particles are never replicates",
                  ranking_rules="complete reference, valid algorithm/NFE/K, verified compatibility; no averaging settings",
                  missing_and_failed_runs="retained in run_status/protocols; never silently imputed")
    write_json(destination / "analysis.json", report)
    if make_plots:
        plots(aggregates, destination / "plots")
        pareto_plots(pooled_fronts, destination / "pareto_plots")
    return report
