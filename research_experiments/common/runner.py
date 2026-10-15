"""One job at a time, load-only execution and post hoc evaluation."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import numpy as np
from .config import input_path
from .cost import Cost, CountedScore
from .distributions import generate, ExactField, Schedule, reference_modes
from .storage import fingerprint, array_hash, file_hash, directory_hash, JobStore
from .initialization import initialize
from .search import run_search
from .metrics import evaluate_modes
from .diagnostics import memory_usage, trajectory_diagnostics, split_diagnostics


def checkpoint_path(c, overrides):
    value = c["checkpoint"]
    if value:
        value = value.format(dim=c["distribution"]["dim"], components=c["distribution"]["components"],
                             distribution_seed=c["distribution_seed"], model_seed=c["model_seed"])
    return input_path(value, overrides)


def saved_starts_path(c, overrides):
    value = c["initialization"]["saved_path"]
    if value:
        value = value.format(seed=c["seed"], model_seed=c["model_seed"], dim=c["distribution"]["dim"],
            components=c["distribution"]["components"], distribution_seed=c["distribution_seed"])
    return input_path(value, overrides)


def inputs(c, overrides, hash_files=False):
    paths = []
    if c["checkpoint"]:
        paths.append(checkpoint_path(c, overrides))
    if c["initialization"]["saved_path"]:
        start = saved_starts_path(c, overrides)
        paths.extend([start, start.with_suffix(".json")])
    if c["task"] == "mnist":
        for k in ["evaluator_checkpoint", "second_evaluator_checkpoint"]:
            if c["mnist"][k]:
                paths.append(input_path(c["mnist"][k], overrides))
        paths.append(input_path(c["mnist"]["data_root"], overrides))
    if c["task"] == "retrospective":
        paths.extend(input_path(p, overrides) for p in c["retrospective"]["roots"])
        if overrides.get("archive_root"):
            paths.append(Path(overrides["archive_root"]).expanduser().resolve())
        if c["retrospective"]["manifest"]:
            paths.append(input_path(c["retrospective"]["manifest"], overrides))
    return [dict(path=str(p), exists=p.exists(), is_file=p.is_file(),
        sha256=(file_hash(p) if p.is_file() else directory_hash(p)) if hash_files and p.exists() else None) for p in paths]


def estimate(c):
    if c["task"] == "retrospective":
        return dict(neural_nfe=0, inference=False,
            warning="Archive reads, optional clustering and numerical reference verification only; no model loading.")
    m, p = c["method"], c["population"]
    steps = sum(m["steps"]) if isinstance(m["steps"], list) else len(m["levels"]) * m["steps"]
    sampling = (c["model"]["T"] if c["initialization"]["kind"] == "ddpm" else
                c["initialization"]["ddim_steps"] if c["initialization"]["kind"] == "ddim" else 0) * p
    movement = 0 if m["name"] == "sampling" else p * (steps + m["final_steps"])
    initial_kind = "neural" if c["initialization"]["source"] == "learned" or c["field"] == "learned" else "analytic"
    main_kind = "neural" if c["field"] == "learned" else "analytic"
    counts = dict(neural_nfe=0, analytic_evaluations=0)
    counts["neural_nfe" if initial_kind == "neural" else "analytic_evaluations"] += sampling
    counts["neural_nfe" if main_kind == "neural" else "analytic_evaluations"] += movement
    return dict(initialization_evaluations=sampling, initialization_unit=initial_kind,
        no_population_change=counts, budget_unit=main_kind,
        warning="Estimate excludes Split probes, extra starts, historical adaptive grids, diagnostics, references and clustering; not an upper bound.",
        quadratic_clustering=m["merge"]["when"] != "none" or m["selection"]["kind"] != "none",
        learned_reference_evaluation=c["task"] == "mnist", budget=c["budget"])


def run_job(job, output, version, overrides, resume=False):
    c = job["config"]
    root = Path(output) / c["suite"] / c["phase"]
    dependencies = inputs(c, overrides, hash_files=True)
    context = dict(config=c, code_sha256=version["source_sha256"], dependencies=dependencies, overrides=overrides,
                   runtime_dependencies=version.get("dependencies"))
    context_hash = fingerprint(context)
    with JobStore(root, job["id"], resume=resume) as store:
        if (store.path / "metadata.json").exists():
            old_meta = json.loads((store.path / "metadata.json").read_text())
            if old_meta["context_sha256"] != context_hash:
                raise ValueError("Resume refused: code/inputs/config changed; use another output_root")
        if store.done:
            store.verify()
            old = json.loads((store.path / "metadata.json").read_text())
            if old["context_sha256"] != context_hash:
                raise ValueError("Resume refused: code/inputs/config changed; use another output_root")
            return "already_complete"
        meta = dict(job_id=job["id"], label=job["label"], context_sha256=context_hash,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            version=version, inputs=dependencies, overrides=overrides, phase=c["phase"],
            old_models_are_fixed=bool(c["checkpoint"]), statistical_unit="mixture/model and search seed, not particle")
        store.json("metadata.json", meta)
        store.json("resolved_config.json", c)
        try:
            if c["task"] == "retrospective":
                from .retrospective import analyze_archives
                roots = [input_path(p, overrides) for p in c["retrospective"]["roots"]]
                if overrides.get("archive_root"):
                    roots.append(Path(overrides["archive_root"]).expanduser())
                report = analyze_archives(roots, c["retrospective"], input_path(c["retrospective"]["manifest"], overrides))
                store.json("retrospective.json", report)
                store.json("summary.json", dict(task="retrospective", n_recomputed=len(report["recomputed"]),
                    n_saved=len(report["saved_records"]), available_methods=report["available_methods"],
                    status="raw_inputs_missing" if not report["recomputed"] else "complete"))
                store.complete()
                return "complete"
            missing = [p["path"] for p in dependencies if not p["exists"]]
            if missing:
                raise FileNotFoundError("Required inputs are absent; no model was run: " + ", ".join(missing))
            begin = time.perf_counter()
            mixture = generate(c["distribution"], c["distribution_seed"]) if c["task"] == "synthetic" else None
            field, learned = None, None
            if c["field"] == "learned" or c["initialization"]["source"] == "learned" or c["evaluation"]["trajectory_alignment"]:
                from .models import LearnedField
                cp = checkpoint_path(c, overrides)
                if cp is None or not cp.is_file():
                    raise FileNotFoundError(f"Required checkpoint missing: {cp}; training is disabled")
                if c["task"] == "synthetic":
                    if c["distribution"]["family"] != "legacy_random":
                        raise ValueError("Existing learned checkpoints are compatible only with legacy_random distributions")
                    expected = f"model_dim{c['distribution']['dim']}_K{c['distribution']['components']}_dseed{c['distribution_seed']}.pth"
                    if cp.name != expected:
                        raise ValueError(f"Cannot verify checkpoint/distribution identity; expected {expected}")
                learned = LearnedField(cp, c["model"], c["task"] == "mnist", c["device"])
                if c["task"] == "mnist" and learned.state.get("config", {}).get("seed") != c["model_seed"]:
                    raise ValueError("MNIST checkpoint training seed differs from model_seed")
                if c["task"] == "synthetic" and learned.dim != mixture.dim:
                    raise ValueError("Checkpoint dimension differs")
            field = learned if c["field"] == "learned" else ExactField(mixture, learned.schedule if learned else Schedule(**{
                k: c["model"][k] for k in ["T", "beta_start", "beta_end"]}))
            cost = Cost(limit=c["budget"], budget_kind=field.kind)
            score = CountedScore(field, cost)
            exact = CountedScore(ExactField(mixture, field.schedule), cost) if mixture else None
            learned_score = CountedScore(learned, cost) if learned else None
            source_score = learned_score if c["initialization"]["source"] == "learned" else score
            source_identity = (dict(checkpoint_sha256=file_hash(checkpoint_path(c, overrides)), model=c["model"])
                if source_score.kind == "neural" else dict(mixture=mixture.as_dict(), schedule=c["model"]))
            source_identity.update(code_sha256=version["source_sha256"],
                device=getattr(source_score.field, "device", "cpu"), dependencies=version.get("dependencies"))
            initialization_started = time.perf_counter()
            initial, init_meta = initialize(source_score, c, Path(output) / "starts", source_identity, mixture,
                external_start=saved_starts_path(c, overrides))
            initialization_seconds = time.perf_counter() - initialization_started
            initial_base = initial.copy()
            init_cost = dict(neural_nfe=cost.neural_nfe, analytic_evaluations=cost.analytic_evaluations)
            start_time = time.perf_counter()
            def replenish(j):
                return initialize(source_score, c, Path(output) / "starts", source_identity, mixture,
                                  n=1, seed=c["seed"] + 90001 + j)[0]
            if c["method"]["name"].startswith("historical_"):
                from .historical import run_legacy
                result = run_legacy(score, initial, c["method"])
            else:
                result = run_search(score, initial, c["method"], replenish, c["seed"])
            movement_seconds = time.perf_counter() - start_time
            search_seconds = initialization_seconds + movement_seconds
            store.json("implementation.json", dict(
                historical_parameters=result.get("historical_parameters"), warning=result.get("warning"),
                field_dimension=field.dim, actual_device=getattr(field, "device", "cpu"),
                coordinate_dtype="NumPy float64; neural score float32",
                historical_internal_stopping_rules_preserved=c["method"]["name"].startswith("historical_")))
            algorithm_cost = dict(neural_nfe=cost.neural_nfe, analytic_evaluations=cost.analytic_evaluations)
            store.arrays("candidates.npz", candidates=result["candidates"], initial=initial_base,
                         all_initial=result["initial"])
            # Ragged trajectories use separate numeric arrays, never object NPZ.
            trace_arrays = {f"x_{i:05d}": s["x"] for i, s in enumerate(result["snapshots"])}
            for s in result["snapshots"]:
                s["coordinate_mode"] = c["method"]["coordinate_mode"]
            trace_meta = [{k: v for k, v in s.items() if k != "x"} for s in result["snapshots"]]
            store.arrays("trajectory.npz", **trace_arrays)
            store.json("trajectory.json", trace_meta)
            store.json("movement.json", result["movement"])
            store.json("stages.json", result["stages"])
            store.json("split_events.json", result["split_events"])
            store.json("initialization.json", init_meta)
            pair_key = fingerprint(dict(task=c["task"], distribution=c["distribution"],
                distribution_seed=c["distribution_seed"], model_seed=c["model_seed"], seed=c["seed"],
                population=c["population"], initialization=c["initialization"], phase=c["phase"]))
            summary = dict(task=c["task"], label=job["label"], job_id=job["id"], suite=c["suite"], phase=c["phase"],
                method=c["method"]["name"], field=c["field"], seed=c["seed"], model_seed=c["model_seed"],
                distribution_seed=c["distribution_seed"], dimension=field.dim, family=c["distribution"]["family"] if mixture else "mnist",
                population=c["population"], n_candidates=len(result["candidates"]), pair_key=pair_key,
                initial_hash=array_hash(initial_base), initialization_cost=init_cost,
                algorithm_cost=algorithm_cost, search_seconds=search_seconds,
                initialization_seconds=initialization_seconds, movement_seconds=movement_seconds,
                initialization_reused=init_meta["reused"],
                timing_scope="Actual initialization/cache access plus method execution; excludes loading and evaluation",
                budget_exhausted=result["budget_exhausted"], budget_unit=field.kind, memory=memory_usage(),
                budget_limit=c["budget"], unused_budget=c["budget"] - cost.budget_used if c["budget"] is not None else None,
                resolved_device=getattr(field, "device", "cpu"),
                requested_k=c["method"]["selection"]["k"],
                k_constraint_met=(len(result["candidates"]) == c["method"]["selection"]["k"])
                    if c["method"]["selection"]["k"] is not None else None,
                population_path_length_sum=float(sum(np.sum(r["actual_l2"]) for r in result["movement"])),
                path_note="Sum over active particle updates; not an individual ancestral path after Split/Merge.")
            eval_time = time.perf_counter()
            if mixture:
                # Only now instantiate modes, after the algorithm has returned.
                with cost.time("evaluation_reference_modes"):
                    refs, reference_status = reference_modes(mixture, restarts=c["evaluation"]["reference_restarts"])
                store.arrays("reference.npz", modes=refs)
                store.json("distribution.json", mixture.as_dict())
                store.json("reference.json", reference_status)
                summary["reference_complete"] = reference_status["complete"]
                summary["metrics"] = evaluate_modes(refs, result["candidates"], c["evaluation"]["radii"], reference_status["rare_mask"])
                residual = score(result["candidates"], c["method"]["final_t"], diagnostic=True)
                summary["stationarity_fraction"] = float(np.mean(np.linalg.norm(residual, axis=1) <= c["evaluation"]["stationarity_tolerance"])) if len(residual) else None
                store.json("split_diagnostics.json", split_diagnostics(result, refs, c["evaluation"]["radii"]))
                if c["evaluation"]["trajectory_alignment"] or c["evaluation"]["basin_diagnostics"]:
                    with cost.time("trajectory_diagnostics"):
                        diag = trajectory_diagnostics(result, exact, learned_score,
                            refs, c["evaluation"]["basin_diagnostics"])
                    store.json("alignment.json", diag)
            else:
                from .mnist import evaluate_mnist
                diagnostic, image_arrays = evaluate_mnist(result, learned, c, output, overrides)
                store.json("mnist.json", diagnostic)
                store.arrays("mnist.npz", **image_arrays)
                summary["mnist"] = diagnostic["stages"][-1]
            summary.update(cost=asdict(cost), model_forward_calls=learned.forward_calls if learned else 0,
                           evaluation_seconds=time.perf_counter() - eval_time,
                           total_seconds=time.perf_counter() - begin)
            store.json("summary.json", summary)
            store.complete()
            return "complete"
        except BaseException as exc:
            store.json("ERROR.json", dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
            raise
