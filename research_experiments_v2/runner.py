"""Resume-safe V2 execution. No training, downloading or automatic confirmation."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import platform
import time
import traceback

import numpy as np

from . import config as configuration
from .storage import (RESULTS, FORMAT, file_hash, fingerprint, array_hash, input_path, output_path,
                      source_identity, runtime_identity, write_json, write_arrays,
                      job_claim, completed, completion)
from .cost import Ledger, CountedScore
from .initialization import StartFactory, noise_rows
from .engine import run, output_candidates
from .problems import make_problem, ExactSchedule
from .evaluation import evaluate_synthetic, score_identity, compatibility, alignment_diagnostics


def resolve(studies, domain=None, device=None, no_cache=False, labels=None):
    groups = []
    for path in studies:
        jobs, contrasts, digest = configuration.load(path)
        selected = []
        for job in jobs:
            c = deepcopy(job["config"])
            if domain and c["domain"] != domain or labels and job["label"] not in labels:
                continue
            if device:
                c["device"] = device
            if no_cache:
                c["initialization"]["cache"] = False
            configuration.validate(c)
            selected.append(dict(job, config=c, id=f"{job['label']}-{fingerprint(c)[:20]}"))
        if selected:
            groups.append(dict(path=str(path), jobs=selected, contrasts=contrasts, config_digest=digest))
    if not groups:
        raise ValueError("No jobs selected")
    ids = [j["id"] for g in groups for j in g["jobs"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate jobs across studies")
    return groups


def plan_summary(groups, overrides=None):
    jobs = [j for g in groups for j in g["jobs"]]
    return dict(format="dms_v2_dry_run", n_jobs=len(jobs),
                by_study=dict(Counter(j["config"]["study"] for j in jobs)),
                by_domain=dict(Counter(j["config"]["domain"] for j in jobs)),
                by_phase=dict(Counter(j["config"]["phase"] for j in jobs)),
                by_dimension=dict(Counter(str(j["config"]["problem"]["dim"]) for j in jobs)),
                query_cap_sum=sum(j["config"]["budget"]["limit"] for j in jobs),
                query_cap_is_not_wall_time_estimate=True,
                outputs_per_job=sorted({len(j["config"]["outputs"]) for j in jobs}),
                contrasts=[dict(study=Path(g["path"]).name, count=len(g["contrasts"])) for g in groups],
                input_files=configuration.required_inputs(jobs, overrides),
                output_root=str(RESULTS), no_model_loaded=True, no_score_evaluated=True)


def doctor(groups, overrides=None, inspect=False):
    jobs = [j for g in groups for j in g["jobs"]]
    neural = any(j["config"]["checkpoint"] for j in jobs)
    legacy = any(j["config"]["method"]["split"]["kind"] != "none" or
                 j["config"]["method"]["implementation"].startswith("historical_") for j in jobs)
    packages = ["numpy", "scipy", "mpmath"] + (["torch"] if neural or legacy else [])
    if any(j["config"]["domain"] == "mnist" for j in jobs):
        packages += ["torchvision"]
    if legacy:
        packages += ["sklearn"]
    missing_packages = [p for p in packages if importlib.util.find_spec(p) is None]
    inputs = configuration.required_inputs(jobs, overrides)
    missing_files = [dict(path=p, purpose=purpose) for p, purpose in inputs.items() if not Path(p).is_file()]
    report = dict(runtime=runtime_identity(), required_packages=packages,
                  missing_packages=missing_packages, missing_files=missing_files,
                  results_guard=str(output_path(RESULTS)), checkpoint_inspection=[],
                  no_training=True, no_download=True, no_score_evaluated=True)
    requested_mps = any(j["config"]["device"] == "mps" for j in jobs)
    report["mps_available"] = None
    if "torch" in packages and "torch" not in missing_packages:
        import torch
        report["mps_available"] = bool(torch.backends.mps.is_available())
    if inspect and not missing_packages and not missing_files:
        import torch
        seen = set()
        for j in jobs:
            c = j["config"]
            if not c["checkpoint"]:
                continue
            p = input_path(c["checkpoint"], overrides)
            if p in seen:
                continue
            seen.add(p)
            state = torch.load(p, map_location="cpu", weights_only=True)
            if c["domain"] == "mnist":
                if state.get("format") != "mnist_diffusion_v1" or state.get("config", {}).get("seed") != c["model_seed"]:
                    raise ValueError(f"MNIST format/model seed mismatch: {p}")
                if any(state.get("model_config", {}).get(k) != c["model"][k] for k in ["T", "beta_start", "beta_end"]):
                    raise ValueError(f"MNIST checkpoint schedule differs from the protocol: {p}")
                if not state.get("split_indices"):
                    raise ValueError(f"Missing MNIST train/validation indices: {p}")
                # Constructor validates data and split indices without inference.
                from src.mnist_experiments.data import ReferenceData
                data = ReferenceData(state, str(input_path(c["mnist"]["data_root"], overrides)), download=False)
                evaluator = torch.load(input_path(c["mnist"]["evaluator_checkpoint"], overrides),
                                       map_location="cpu", weights_only=True)
                if evaluator.get("format") != "mnist_evaluator_v1":
                    raise ValueError("Invalid frozen evaluator format")
                for split in ["train", "validation"]:
                    if not torch.equal(torch.as_tensor(evaluator["metadata"][split + "_indices"]).long().cpu(), data.indices[split]):
                        raise ValueError("Evaluator/data splits mismatch")
            elif state.get("dim") != c["problem"]["dim"] or state.get("T") != c["model"]["T"]:
                raise ValueError(f"Synthetic checkpoint shape/schedule mismatch: {p}")
            report["checkpoint_inspection"].append(dict(path=str(p), sha256=file_hash(p),
                                                         format=state.get("format", "legacy_synthetic")))
        report["checkpoint_inspection_note"] = "No forward pass. Legacy synthetic training metadata still requires external evidence."
    report["ready"] = not missing_packages and not missing_files and not (requested_mps and report["mps_available"] is False)
    return report


class Resources:
    def __init__(self, overrides=None, compatibility_evidence=None):
        self.overrides = overrides or {}
        self.source, self.runtime = source_identity(), runtime_identity()
        self.hashes = {}
        self.learned_key, self.learned_field = None, None
        self.mnist_key, self.mnist_evaluator = None, None
        self.compatibility_evidence = compatibility_evidence or {}
        self.reference_cache = {}  # evaluation-only; never attached to score/search

    def digest(self, path):
        p = str(path)
        stat = Path(path).stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        if p not in self.hashes or self.hashes[p][0] != signature:
            self.hashes[p] = (signature, file_hash(path))
        return self.hashes[p][1]

    def field(self, c):
        problem = None if c["domain"] == "mnist" else make_problem(c["problem"], c["distribution_seed"])
        learned = None
        compatibility_report = None
        input_hashes = {p: self.digest(p) for p in configuration.required_inputs([dict(config=c)], self.overrides)}
        if c["checkpoint"]:
            from research_experiments.common.models import LearnedField
            p = input_path(c["checkpoint"], self.overrides)
            key = fingerprint(dict(sha=self.digest(p), model=c["model"], device=c["device"], mnist=c["domain"] == "mnist"))
            if key != self.learned_key:
                # Keep one model alive, not every synthetic/model seed on MPS.
                self.learned_field = None
                self.learned_field = LearnedField(p, c["model"], c["domain"] == "mnist", c["device"])
                self.learned_key = key
            learned = self.learned_field
            if learned.dim != c["problem"]["dim"]:
                raise ValueError("Checkpoint dimension does not match problem")
            if c["domain"] == "mnist" and learned.state.get("config", {}).get("seed") != c["model_seed"]:
                raise ValueError("Checkpoint training seed differs from requested model seed")
            if c["domain"] == "alignment":
                compatibility_report = compatibility(learned, problem, c, self.digest(p), self.compatibility_evidence)
        if c["domain"] == "exact":
            field = problem.field(ExactSchedule(**{k: c["model"][k] for k in ["T", "beta_start", "beta_end"]}))
        elif c["domain"] == "alignment":
            from research_experiments.common.distributions import ExactField
            exact = ExactField(problem, learned.schedule)  # actual checkpoint float32 alpha_bar, no clean-zero override
            field = exact if c["field"] == "exact" else learned
        else:
            field = learned
        return problem, field, learned, compatibility_report, input_hashes

    def mnist(self, field, c):
        from .mnist import MNISTEvaluator
        key = fingerprint(dict(config=c["mnist"], checkpoint=c["checkpoint"], device=field.device,
                               hashes={p: self.digest(p) for p in configuration.required_inputs([dict(config=c)], self.overrides)}))
        if key != self.mnist_key:
            self.mnist_evaluator = MNISTEvaluator(field, c, self.overrides)
            self.mnist_key = key
        return self.mnist_evaluator


def recover_stale(directory):
    lock = output_path(directory) / "RUNNING.lock"
    if not lock.exists():
        return
    data = json.loads(lock.read_text())
    if data["host"] != platform.node():
        raise RuntimeError("Lock belongs to another host; inspect it before removing")
    try:
        os.kill(data["pid"], 0)
    except ProcessLookupError:
        lock.unlink()
        return
    raise RuntimeError("Recorded process is still alive; not reclaiming its job")


def execute_job(job, group, resources, base=RESULTS / "pilot", recover=False):
    c = job["config"]
    problem, field, learned, comp, hashes = resources.field(c)
    # Exact/learned alignment shares the same execution context, including its
    # neural initializer. Field-specific cost counters still remain distinct.
    device = getattr(learned, "device", getattr(field, "device", "numpy_cpu"))
    cohort = fingerprint(dict(source=resources.source["digest"], runtime=resources.runtime, device=device))[:20]
    identity = fingerprint(dict(config=c, inputs=hashes, source=resources.source["digest"],
                                runtime=resources.runtime, device=device, compatibility=comp))
    directory = output_path(Path(base) / cohort / c["study"] / job["id"])
    if completed(directory, identity):
        return "resumed", directory
    if (directory / "DONE.json").exists():
        raise RuntimeError(f"Completed job identity/integrity changed; choose a new run root: {directory}")
    if recover:
        recover_stale(directory)
    with job_claim(directory):
        try:
            write_json(directory / "resolved_config.json", c)
            metadata = dict(format=FORMAT, job_id=job["id"], label=job["label"], cohort=cohort,
                            source=resources.source, runtime=resources.runtime, device=device,
                            field_device=getattr(field, "device", "numpy_cpu"),
                            input_sha256=hashes, study_config=group["path"], study_config_digest=group["config_digest"],
                            compatibility=comp, phase=c["phase"], result_origin="new_v2_execution")
            write_json(directory / "metadata.json", metadata)
            ledger = Ledger(c["budget"]["limit"])
            score = CountedScore(field, ledger)
            init_field = learned if c["initialization"]["source"] == "learned" else field
            init_score = CountedScore(init_field, ledger)
            init_identity = dict(density=None if problem is None else score_identity(problem),
                                 checkpoint=hashes.get(str(input_path(c["checkpoint"], resources.overrides))) if c["checkpoint"] else None,
                                 source=resources.source["digest"], runtime=resources.runtime,
                                 device=getattr(init_field, "device", "numpy_cpu"))
            starts = StartFactory(init_score, c["seed"], c["initialization"]["steps"], init_identity,
                                  c["domain"] == "mnist", c["initialization"]["cache"], c["population"])
            forward_before = learned.forward_calls if learned else 0
            begin = time.perf_counter()
            result = run(score, starts, c["method"], c["budget"], c["population"])
            search_seconds = time.perf_counter() - begin
            write_json(directory / "implementation.json", dict(
                implementation=c["method"]["implementation"], historical_parameters=result.get("historical_parameters"),
                warning=result.get("warning"), zero_convention="clean_zero" if c["domain"] == "exact" else "trained_indexed_zero",
                dtype="NumPy float64 state; existing neural model float32",
                history="original class with shared external starts" if c["method"]["implementation"].startswith("historical_")
                        else "v2 coordinator; v1 movement and geometry kernels; original local Split when requested"))
            outputs = {}
            with ledger.scope("output_geometry"):
                for policy in c["outputs"]:
                    outputs[policy["name"]] = output_candidates(result["candidates"], result["lineage"], policy)
            nfe_valid = (ledger.budget_used <= c["budget"]["limit"] and
                         (c["budget"]["regime"] == "cap" or ledger.remaining <= c["budget"]["tolerance"]))
            algorithm_valid = not (c["method"]["implementation"].startswith("historical_") and not result["completed_plan"])
            common = dict(format=FORMAT, job_id=job["id"], label=job["label"], cohort=cohort,
                          source_digest=resources.source["digest"],
                          domain=c["domain"], field=c["field"], study=c["study"], phase=c["phase"],
                          dimension=c["problem"]["dim"], family=c["problem"]["family"],
                          distribution_seed=c["distribution_seed"], search_seed=c["seed"], model_seed=c["model_seed"],
                          treatment=configuration.treatment(c), P0=c["population"],
                          P_total=result["n_starts_generated"], budget=c["budget"]["limit"],
                          regime=c["budget"]["regime"], nfe_tolerance=c["budget"]["tolerance"],
                          queries=ledger.budget_used, neural_nfe=ledger.neural_nfe, analytic_evaluations=ledger.analytic_evaluations,
                          unused_budget=ledger.remaining, nfe_valid=nfe_valid, algorithm_valid=algorithm_valid,
                          search_seconds=search_seconds, search_model_forward_calls=(learned.forward_calls - forward_before) if learned else 0,
                          common_noise_sha256=array_hash(noise_rows(c["seed"], range(c["population"]), field.dim)),
                          common_initial_sha256=array_hash(result["initial"][:c["population"]]),
                          density_identity=None if problem is None else score_identity(problem),
                          checkpoint_sha256=init_identity["checkpoint"], initialization_steps=c["initialization"]["steps"],
                          zero_convention="clean_zero" if c["domain"] == "exact" else "trained_indexed_zero",
                          compatibility_verified=bool(comp["verified_training_protocol"]) if comp else True)
            evaluation_begin = time.perf_counter()
            rows = []
            if c["domain"] == "mnist":
                report, arrays = resources.mnist(field, c).evaluate(result, outputs)
                write_json(directory / "mnist.json", report)
                write_arrays(directory / "mnist.npz", **arrays)
                metric_rows = {name: [dict(v, radius=None, reference_complete=None)] for name, v in report["outputs"].items()}
            else:
                ref_key = score_identity(problem)
                catalog = resources.reference_cache.get(ref_key)
                metric_rows, report, arrays = evaluate_synthetic(problem, result, outputs, c["evaluation"]["radii"], c["evaluation"]["basin_diagnostics"], catalog)
                resources.reference_cache[ref_key] = (arrays["modes"], report["reference"])
                write_json(directory / "reference.json", report["reference"])
                write_arrays(directory / "reference.npz", **arrays)
                write_json(directory / "basin_interventions.json", report)
                common["cross_clean_basin_groups"] = (sum(e["cross_clean_basin_groups"] for e in report["merge"])
                                                      if report.get("certified_clean_basin_diagnostics") and report["intervention_events_observed"] else None)
                common["groups_with_certified_basins"] = sum(e["groups_with_certified_basins"] for e in report["merge"])
                direct = [v for e in report["merge"] for v in e["direct_mode_metrics"] if v["radius"] == .25]
                observed = report["intervention_events_observed"] and report["reference"]["complete"] and .25 in c["evaluation"]["radii"]
                common["merge_near_mode_cross_groups"] = sum(v["near_mode_cross_groups"] for v in direct) if observed else None
                common["merge_event_lost_matches"] = sum(v["event_lost_matches"] for v in direct) if observed else None
                if c["domain"] == "alignment" and c["evaluation"]["alignment"]:
                    from research_experiments.common.distributions import ExactField
                    exact = ExactField(problem, learned.schedule)
                    alignment, arrays = alignment_diagnostics(exact, learned, result, arrays["modes"], ledger,
                                                             c["evaluation"]["diagnostic_points"], c["seed"])
                    write_json(directory / "alignment.json", alignment)
                    write_arrays(directory / "alignment.npz", **arrays)
            for name, values in metric_rows.items():
                out = outputs[name]
                for metrics in values:
                    row = dict(common, **metrics)
                    row.update(output_policy=name, output_kind=out["policy"]["kind"], requested_k=out["policy"]["k"],
                               k_constraint_met=out["k_constraint_met"], output_geometry=fingerprint(out["policy"]),
                               n_surviving_unique=out["n_surviving_unique"], K=len(out["candidates"]))
                    row["ranking_eligible"] = bool(nfe_valid and algorithm_valid and out["k_constraint_met"] and
                                                   (c["domain"] == "mnist" or row["reference_complete"]) and
                                                   row["compatibility_verified"])
                    rows.append(row)
            write_json(directory / "metrics.json", rows)
            write_json(directory / "cost.json", ledger.as_dict())
            write_json(directory / "timing.json", dict(search_seconds=search_seconds,
                evaluation_seconds=time.perf_counter() - evaluation_begin,
                initialization_cache_reused=any(r["reused"] for r in starts.records),
                search_timing_scope="initialization/cache access + search only; loading/reference evaluation excluded",
                wall_time_comparison="disable initialization caches for comparable execution timing",
                diagnostic_model_forward_calls=(learned.forward_calls - forward_before - common["search_model_forward_calls"]) if learned else 0))
            write_json(directory / "initialization.json", dict(records=starts.records, prefix_count=c["population"],
                                                                  noise_sha256=common["common_noise_sha256"], initial_sha256=common["common_initial_sha256"]))
            write_arrays(directory / "candidates.npz", initial=result["initial"], final_state=result["candidates"],
                         **{"output_" + name: out["candidates"] for name, out in outputs.items()})
            # Ragged arrays go to independent numeric NPZ keys, never pickled object arrays.
            write_arrays(directory / "trajectory.npz", **{f"x{i:04d}": s["x"] for i, s in enumerate(result["snapshots"])})
            write_json(directory / "trajectory.json", [{k: v for k, v in s.items() if k != "x"} for s in result["snapshots"]])
            for key in ["movement", "stages", "split_events", "merge_events", "restart_events"]:
                write_json(directory / (key + ".json"), result[key])
            write_json(directory / "status.json", {k: result[k] for k in ["completed_plan", "stopped_by_budget", "n_starts_generated"]})
            completion(directory, identity)
            return "done", directory
        except Exception as exc:
            write_json(directory / "ERROR.json", dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
            raise


def run_groups(groups, overrides=None, base=RESULTS / "pilot", compatibility_evidence=None,
               recover=False, fail_fast=False, confirmation_lock=None):
    phases = {j["config"]["phase"] for g in groups for j in g["jobs"]}
    if "confirmation" in phases:
        if not confirmation_lock or phases != {"confirmation"}:
            raise ValueError("Confirmation requires a validated lock, and cannot be mixed with pilot")
        if fingerprint(groups) != fingerprint(confirmation_lock["groups"]):
            raise ValueError("Resolved jobs differ from the validated confirmation lock")
    preflight = doctor(groups, overrides, inspect=True)
    if not preflight["ready"]:
        raise RuntimeError("Preflight failed:\n" + json.dumps(preflight, ensure_ascii=False, indent=2))
    resources = Resources(overrides, compatibility_evidence)
    base = output_path(base)
    write_json(base / "run_plan.json", dict(summary=plan_summary(groups, overrides), preflight=preflight,
                                              source=resources.source, runtime=resources.runtime,
                                              studies=[{k: v for k, v in g.items() if k != "jobs"} for g in groups]))
    failures, counts = [], Counter()
    for group in groups:
        write_json(base / "protocols" / (Path(group["path"]).stem + ".json"), group)
        for job in group["jobs"]:
            try:
                status, directory = execute_job(job, group, resources, base, recover)
                counts[status] += 1
                print(f"{status}: {job['id']} ({directory})", flush=True)
            except Exception as exc:
                failures.append(dict(job=job["id"], error=str(exc)))
                print(f"ERROR: {job['id']}: {exc}", flush=True)
                if fail_fast:
                    raise
    write_json(base / "run_status.json", dict(counts=dict(counts), failures=failures))
    return dict(counts=dict(counts), failures=failures)
