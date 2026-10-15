"""Aggregate saved jobs, paired contrasts, uncertainty and separate Pareto axes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from ..common.storage import REPO, safe_path, fingerprint, write_json, JobStore
from ..common.statistics import hierarchical_interval
from ..common.metrics import pareto_mask
from .saved_tables import archived_tables


def write_csv(path, rows):
    p = safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()})


def load_rows(root):
    rows = []
    for p in sorted(Path(root).rglob("DONE.json")):
        store = JobStore(p.parent.parent, p.parent.name, resume=True)
        store.verify()
        c = json.loads((p.parent / "resolved_config.json").read_text())
        s = json.loads((p.parent / "summary.json").read_text())
        if s["task"] == "retrospective":
            report = json.loads((p.parent / "retrospective.json").read_text())
            for r in report["recomputed"]:
                scalar = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                rows.append(dict(scalar, label=r["method"], field="historical", phase="retrospective",
                    neural_nfe=r["cost"]["neural_nfe"], analytic_evaluations=r["cost"]["analytic_evaluations"],
                    reference_complete=r["reference_status"]["complete"], suite="00_retrospective",
                    setting=r["method"] + "-" + str(r["dimension"]),
                    cluster=str(r.get("distribution_seed")), pair_key=None, initial_hash=None))
            continue
        metadata = json.loads((p.parent / "metadata.json").read_text())
        calibration = json.loads((p.parent / "mnist.json").read_text())["reference_key"] if s["task"] == "mnist" else None
        cohort = fingerprint(dict(code=metadata["version"]["source_sha256"],
            environment=metadata["version"].get("dependencies"), calibration=calibration))
        setting = fingerprint(dict(label=s["label"], method=c["method"], distribution=c["distribution"],
            population=c["population"], initialization=c["initialization"], budget=c["budget"],
            field=c["field"], task=c["task"], suite=c["suite"], model=c["model"], evaluation=c["evaluation"],
            mnist_evaluation=c["mnist"] if c["task"] == "mnist" else None, cohort=cohort))[:16]
        base = {k: s[k] for k in ["label", "job_id", "phase", "suite", "family", "dimension", "population", "seed", "model_seed",
            "distribution_seed", "field", "n_candidates", "pair_key", "initial_hash", "search_seconds", "budget_exhausted"]}
        base.update(setting=setting, neural_nfe=s["algorithm_cost"]["neural_nfe"],
            analytic_evaluations=s["algorithm_cost"]["analytic_evaluations"],
            cluster=str(s["model_seed"] if s["task"] == "mnist" else s["distribution_seed"]),
            reference_complete=s.get("reference_complete"))
        base.update(initialization_reused=s["initialization_reused"], requested_k=s["requested_k"],
                    k_constraint_met=s["k_constraint_met"], unused_budget=s["unused_budget"])
        base.update(cohort=cohort, pair_key=fingerprint([s["pair_key"], cohort]))
        if s["task"] == "mnist":
            rows.append(dict(base, radius=None, **{k: v for k, v in s["mnist"].items() if isinstance(v, (float, int))}))
        else:
            for metric in s["metrics"]:
                rows.append(dict(base, **{k: v for k, v in metric.items() if not isinstance(v, (list, dict))}))
    return rows


METRICS = ["recall", "one_to_one_f1", "soft_error", "localization_error", "rare_recall", "duplicate_fraction",
    "n_candidates", "neural_nfe", "analytic_evaluations", "search_seconds", "pixel_knn", "feature_knn", "pixel_support", "feature_support",
    "class_entropy", "class_coverage", "class_change_fraction", "within_class_rms", "near_duplicate_fraction"]


def aggregate(rows, repeats=2000):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["setting"], r["phase"], r.get("radius"))].append(r)
    out = []
    for (setting, phase, radius), data in groups.items():
        for metric in METRICS:
            by_cluster = defaultdict(list)
            for r in data:
                value = r.get(metric)
                if isinstance(value, (float, int)) and np.isfinite(value):
                    by_cluster[r["cluster"]].append(value)
            if not by_cluster:
                continue
            out.append(dict(setting=setting, label=data[0]["label"], suite=data[0]["suite"], phase=phase, radius=radius,
                dimension=data[0]["dimension"], family=data[0].get("family"), field=data[0]["field"], metric=metric,
                reference_complete=all(r.get("reference_complete") for r in data),
                **hierarchical_interval(list(by_cluster.values()), repeats=repeats)))
    return out


def paired_contrasts(rows, contrasts, repeats=2000):
    out = []
    # Familywise intervals for ALL requested contrasts, settings, metrics, radii.
    pending = []
    for a, b in contrasts:
        aa, bb = [r for r in rows if r["label"] == a], [r for r in rows if r["label"] == b]
        if not aa or not bb:
            raise ValueError(f"Missing labels for contrast {a}:{b}")
        settings_a, settings_b = sorted({r["setting"] for r in aa}), sorted({r["setting"] for r in bb})
        for sa in settings_a:
            for sb in settings_b:
                left, right = [r for r in aa if r["setting"] == sa], [r for r in bb if r["setting"] == sb]
                index = {(r["pair_key"], r.get("radius")): r for r in right if r["pair_key"] is not None}
                paired = [(r, index[(r["pair_key"], r.get("radius"))]) for r in left
                    if r["pair_key"] is not None and (r["pair_key"], r.get("radius")) in index]
                if not paired:
                    continue
                if any(x["initial_hash"] != y["initial_hash"] for x, y in paired):
                    raise ValueError("Paired comparison requested for nonidentical starts")
                for radius in sorted({x.get("radius") for x, _ in paired}, key=lambda x: -1 if x is None else x):
                    for metric in METRICS:
                        by_cluster = defaultdict(list)
                        selected = [(x, y) for x, y in paired if x.get("radius") == radius]
                        for x, y in selected:
                            if all(isinstance(v, (float, int)) and np.isfinite(v) for v in [x.get(metric), y.get(metric)]):
                                by_cluster[x["cluster"]].append(x[metric] - y[metric])
                        if by_cluster:
                            pending.append((dict(contrast=f"{a}-{b}", setting_a=sa, setting_b=sb, metric=metric,
                                radius=radius, phase=selected[0][0]["phase"], paired=True), list(by_cluster.values())))
    for info, groups in pending:
        out.append(dict(info, **hierarchical_interval(groups, repeats=repeats, alpha=.05 / max(len(pending), 1))))
    return out


def plots(rows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pairs = [("neural_nfe", "recall"), ("analytic_evaluations", "recall"),
        ("neural_nfe", "one_to_one_f1"), ("analytic_evaluations", "one_to_one_f1"),
        ("n_candidates", "recall"), ("dimension", "duplicate_fraction"),
        ("localization_error", "recall"), ("neural_nfe", "rare_recall"),
        ("n_candidates", "duplicate_fraction"), ("search_seconds", "recall"),
        ("neural_nfe", "pixel_knn"), ("neural_nfe", "feature_knn")]
    # Every facet keeps dimension, phase, radius, family and field separate.
    facets = defaultdict(list)
    for r in rows:
        facets[(r["suite"], r["phase"], r.get("family"), r.get("dimension"), r.get("radius"), r["field"])].append(r)
    for facet, data in facets.items():
        for xkey, ykey in pairs:
            good = [r for r in data if all(isinstance(r.get(k), (float, int)) and np.isfinite(r[k]) for k in [xkey, ykey])]
            if not good or not any(r[xkey] != 0 for r in good):
                continue
            fig, ax = plt.subplots(figsize=(6.5, 4.4))
            for label in sorted({r["label"] for r in good}):
                subset = [r for r in good if r["label"] == label]
                ax.scatter([r[xkey] for r in subset], [r[ykey] for r in subset], s=18, alpha=.6, label=label)
            ax.set(xlabel=xkey, ylabel=ykey, title=" | ".join(map(str, facet)))
            ax.legend(fontsize=6)
            fig.tight_layout()
            path = safe_path(Path(output) / f"{fingerprint(facet)[:10]}_{ykey}_vs_{xkey}.png")
            fig.savefig(path, dpi=160)
            plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", help="Saved suite/phase or results root")
    p.add_argument("--output", required=True, help="A NEW subdirectory of research_experiments/results")
    p.add_argument("--contrast", action="append", default=[], help="variant_a:variant_b, identical starts required")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    root = Path(args.root) if Path(args.root).is_absolute() else REPO / args.root
    output = safe_path(Path(args.output) if Path(args.output).is_absolute() else REPO / args.output)
    if output.exists():
        raise FileExistsError("Use a new analysis output directory")
    rows = load_rows(root)
    saved_aggregates, saved_runs, saved_duplicates = archived_tables(root)
    if not rows and not saved_aggregates and not saved_runs:
        raise ValueError("No completed numeric jobs/raw retrospective inputs found. No tables were invented.")
    output.mkdir(parents=True)
    contrasts = [x.split(":") for x in args.contrast]
    if any(len(x) != 2 for x in contrasts):
        raise ValueError("Use --contrast a:b")
    if args.bootstrap < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    write_csv(output / "runs.csv", rows)
    aggregates = aggregate(rows, args.bootstrap)
    write_csv(output / "aggregates.csv", aggregates)
    write_csv(output / "archived_aggregates.csv", saved_aggregates)
    write_csv(output / "archived_per_run.csv", saved_runs)
    write_json(output / "archive_duplicates.json", saved_duplicates)
    write_csv(output / "paired_contrasts.csv", paired_contrasts(rows, contrasts, args.bootstrap))
    frontier = []
    # Separate axes and fields; no weighted overall score or mixed neural/exact NFE.
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["suite"], r["phase"], r.get("family"), r["dimension"], r.get("radius"), r["field"])].append(r)
    for key, data in buckets.items():
        cost = "analytic_evaluations" if key[-1] == "exact" else "neural_nfe"
        good = [r for r in data if all(isinstance(r.get(k), (float, int)) and np.isfinite(r[k]) for k in [cost, "n_candidates", "recall", "one_to_one_f1"])]
        if good:
            mask = pareto_mask([[r[k] for k in [cost, "n_candidates", "recall", "one_to_one_f1"]] for r in good], [True, True, False, False])
            frontier.extend(dict(r, nondominated=bool(m), frontier_scope="descriptive run-level; no dominance inference") for r, m in zip(good, mask))
    write_csv(output / "pareto.csv", frontier)
    # Also form descriptive frontiers from cluster-weighted means, rather than
    # comparing favorable individual random draws as algorithmic evidence.
    mean_rows = {}
    for row in aggregates:
        key = (row["setting"], row["phase"], row["radius"])
        mean_rows.setdefault(key, {k: row[k] for k in ["setting", "label", "suite", "phase", "radius", "dimension", "family", "field"]})
        mean_rows[key][row["metric"]] = row["mean"]
    grouped_means = defaultdict(list)
    for row in mean_rows.values():
        grouped_means[(row["suite"], row["phase"], row["radius"], row["dimension"], row["family"], row["field"])].append(row)
    mean_frontier = []
    for key, data in grouped_means.items():
        unit = "analytic_evaluations" if key[-1] == "exact" else "neural_nfe"
        columns = [unit, "n_candidates", "recall", "one_to_one_f1"]
        good = [r for r in data if all(isinstance(r.get(k), (float, int)) and np.isfinite(r[k]) for k in columns)]
        if good:
            mask = pareto_mask([[r[k] for k in columns] for r in good], [True, True, False, False])
            mean_frontier.extend(dict(r, nondominated=bool(m)) for r, m in zip(good, mask))
    write_csv(output / "pareto_aggregates.csv", mean_frontier)
    write_json(output / "analysis.json", dict(source=str(root), n_rows=len(rows), contrasts=contrasts,
        bootstrap=args.bootstrap, interval_unit="mixture/model then search seed; equal cluster weight",
        multiplicity="Bonferroni across all emitted requested contrasts/settings/metrics/radii",
        incomplete_reference_warning="When reference_complete=false, recall/F1 are conditional on the discovered reference subset."))
    if not args.no_plots and rows:
        plots(rows, output)
    print(output)


if __name__ == "__main__":
    main()
