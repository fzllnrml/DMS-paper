"""Run from the repository root: python -m research_experiments_v2 --help."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def common(parser):
    parser.add_argument("--manifest", default=str(ROOT / "configs" / "pilot_all.json"))
    parser.add_argument("--config", help="One study instead of a manifest")
    parser.add_argument("--domain", choices=["exact", "alignment", "mnist"])
    parser.add_argument("--device", choices=["cpu", "mps", "auto"])
    parser.add_argument("--mnist-root", help="Directory containing mnist/, mnist_seed43/, mnist_seed44/ (default repo/runs)")
    parser.add_argument("--checkpoint-root", help="Directory containing dim_10/, dim_30/, dim_50/")
    parser.add_argument("--data-root", help="Directory containing MNIST/raw/")
    parser.add_argument("--model-paths", help="JSON mapping configured model/evaluator paths to arbitrary external files")
    parser.add_argument("--labels", nargs="+", help="Optional exact labels; never used for an unlocked confirmation")
    parser.add_argument("--no-cache", action="store_true", help="Execute initialization again; intrinsic NFE is unchanged")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["doctor", "dry-run", "run-pilot", "analyze", "lock-confirmation"]:
        p = sub.add_parser(name)
        common(p)
        if name == "doctor":
            p.add_argument("--inspect-checkpoints", action="store_true", help="Load metadata/data splits only; no forward pass")
        if name == "dry-run":
            p.add_argument("--write-plan", help="Optional JSON path inside v2/results/")
        if name == "run-pilot":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--run-root", default=str(RESULTS / "pilot"))
            p.add_argument("--compatibility-manifest", help="Optional training evidence keyed by checkpoint SHA256")
            p.add_argument("--recover-stale", action="store_true")
            p.add_argument("--fail-fast", action="store_true", help="Default continues remaining jobs and returns failure status at the end")
        if name == "analyze":
            p.add_argument("--root", default=str(RESULTS / "pilot"))
            p.add_argument("--out", default=str(RESULTS / "analysis_pilot"))
            p.add_argument("--bootstrap", type=int, default=5000)
            p.add_argument("--primary-bootstrap", type=int, default=100000)
            p.add_argument("--plots", action="store_true")
            p.add_argument("--lock", help="Analyze exactly the selected locked confirmation contrasts")
        if name == "lock-confirmation":
            p.add_argument("--selection", required=True)
            p.add_argument("--out", default=str(RESULTS / "protocols" / "confirmation_lock.json"))
    p = sub.add_parser("propose-selection")
    p.add_argument("--root", default=str(RESULTS / "pilot"))
    p.add_argument("--out", default=str(RESULTS / "protocols" / "selection_draft.json"))
    p = sub.add_parser("run-confirmation")
    p.add_argument("--lock", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--recover-stale", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p = sub.add_parser("audit-v1")
    p.add_argument("--archive", required=True)
    p.add_argument("--out", default=str(RESULTS / "v1_archive_audit"))
    p = sub.add_parser("export-v1-evidence")
    p.add_argument("--old-results-root", required=True)
    p.add_argument("--request", default=str(ROOT / "docs" / "v1_evidence_request.json"))
    p.add_argument("--out", default=str(RESULTS / "exports" / "v1_evidence.zip"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--suites", nargs="+", help="Optional original suite names, e.g. 08_exact_vs_learned 09_mnist_diagnostics")
    p.add_argument("--tier", choices=["metadata_only", "raw_priority"])
    p = sub.add_parser("verify-isolation")
    p.add_argument("--commit", default="10454bee79847a1246662fa5c280aba8b71b6801")
    p.add_argument("--baseline", help="Optional earlier full local snapshot, including old results")
    p = sub.add_parser("snapshot-isolation")
    p.add_argument("--out", default=str(RESULTS / "audit" / "isolation_baseline.json"))
    args = parser.parse_args(argv)
    if args.command == "doctor":
        # A fresh empty venv must produce a useful diagnosis before importing
        # NumPy/SciPy-dependent protocol modules.
        import importlib.util
        missing = [name for name in ["numpy", "scipy", "mpmath"] if importlib.util.find_spec(name) is None]
        if missing:
            print(json.dumps(dict(ready=False, missing_packages=missing, no_model_loaded=True,
                                  no_score_evaluated=True, install="python -m pip install -r research_experiments_v2/requirements.txt"), indent=2))
            return 1
    if args.command in {"audit-v1", "export-v1-evidence"}:
        from .archive_audit import audit, export_evidence
        result = (audit(args.archive, args.out) if args.command == "audit-v1" else
                  export_evidence(args.old_results_root, args.request, args.out, args.dry_run, args.suites, args.tier))
    elif args.command in {"snapshot-isolation", "verify-isolation"}:
        from .isolation import snapshot, verify_git, verify_snapshot
        if args.command == "snapshot-isolation":
            result = snapshot(args.out)
        else:
            result = verify_snapshot(args.baseline) if args.baseline else verify_git(args.commit)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["byte_identical"] else 1
    elif args.command == "propose-selection":
        from .protocol import propose_selection
        result = propose_selection(args.root, args.out)
    elif args.command == "run-confirmation":
        from .protocol import validate_lock
        from .runner import plan_summary, run_groups
        lock = validate_lock(args.lock)
        if args.dry_run:
            result = plan_summary(lock["groups"], lock["overrides"])
        else:
            result = run_groups(lock["groups"], lock["overrides"], RESULTS / "confirmation",
                                recover=args.recover_stale, fail_fast=args.fail_fast, confirmation_lock=lock)
    else:
        from .config import manifest
        from .runner import resolve, plan_summary, doctor, run_groups
        overrides = {k: getattr(args, k) for k in ["mnist_root", "checkpoint_root", "data_root"] if getattr(args, k)}
        if args.model_paths:
            overrides["model_paths"] = json.loads(Path(args.model_paths).expanduser().read_text())
            if not isinstance(overrides["model_paths"], dict) or any(not isinstance(v, str) for v in overrides["model_paths"].values()):
                raise ValueError("--model-paths must contain a path-to-path JSON object")
        paths = [Path(args.config)] if args.config else manifest(args.manifest)[1]
        groups = resolve(paths, args.domain, args.device, args.no_cache, args.labels)
        if args.command == "dry-run" or args.command == "run-pilot" and args.dry_run:
            result = plan_summary(groups, overrides)
            if getattr(args, "write_plan", None):
                from .storage import write_json
                write_json(args.write_plan, dict(summary=result, groups=groups))
        elif args.command == "doctor":
            result = doctor(groups, overrides, args.inspect_checkpoints)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 1
        elif args.command == "run-pilot":
            if any(j["config"]["phase"] == "confirmation" for g in groups for j in g["jobs"]):
                raise ValueError("Use run-confirmation with a validated lock")
            evidence = json.loads(Path(args.compatibility_manifest).read_text()) if args.compatibility_manifest else None
            result = run_groups(groups, overrides, args.run_root, evidence, args.recover_stale, args.fail_fast)
        elif args.command == "analyze":
            from .analysis import analyze
            hypotheses = None
            if args.lock:
                from .protocol import validate_lock
                lock = validate_lock(args.lock)
                groups, hypotheses = lock["groups"], lock["hypotheses"]
            result = analyze(args.root, args.out, groups, args.bootstrap, args.primary_bootstrap, args.plots, hypotheses)
        elif args.command == "lock-confirmation":
            from .protocol import lock_confirmation
            # The default manifest for this command is the prepared blueprint.
            if args.manifest == str(ROOT / "configs" / "pilot_all.json") and not args.config:
                groups = resolve(manifest(ROOT / "configs" / "confirmation_all.json")[1], device=args.device, no_cache=args.no_cache)
            pilots = resolve(manifest(ROOT / "configs" / "pilot_all.json")[1], device=args.device, no_cache=args.no_cache)
            result = lock_confirmation(groups, pilots, args.selection, args.out, overrides)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return int(bool(result.get("failures"))) if isinstance(result, dict) else 0


if __name__ == "__main__":
    raise SystemExit(main())
