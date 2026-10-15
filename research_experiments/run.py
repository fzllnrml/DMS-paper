"""CLI: plan, run, lock, doctor, unlock; no inference on import or dry run."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fnmatch
import importlib.util
import json
import os
from pathlib import Path
import sys

from .common.config import load
from .common.runner import run_job, estimate, inputs
from .common.storage import REPO, safe_path, fingerprint, source_version, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["run", "lock", "doctor"]:
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--device", choices=["cpu", "mps", "auto"])
        for path in ["mnist-root", "checkpoint-root", "data-root", "archive-root"]:
            p.add_argument("--" + path)
        p.add_argument("--job", help="Label or job-id glob; dry-run lists IDs")
        p.add_argument("--max-jobs", type=int)
        if name == "run":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--lock", help="Required for confirmation")
            p.add_argument("--keep-going", action="store_true")
        if name == "lock":
            p.add_argument("--development-root", help="Optional completed development results for provenance")
    p = sub.add_parser("unlock")
    p.add_argument("path", help="Job directory with stale RUNNING file")
    args = parser.parse_args(argv)
    if args.command == "unlock":
        p = safe_path(Path(args.path) if Path(args.path).is_absolute() else REPO / args.path)
        lock = p / "RUNNING"
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
            print("Removed stale lock:", p)
            return 0
        raise RuntimeError(f"PID {pid} still exists; lock was retained")
    overrides = {k: getattr(args, k) for k in ["mnist_root", "checkpoint_root", "data_root", "archive_root"] if getattr(args, k)}
    jobs, output, digest = load(args.config, overrides)
    if args.device:
        for j in jobs:
            j["config"]["device"] = args.device
            # Device is part of execution protocol and results identity.
            j["id"] = j["label"] + "-" + fingerprint(j["config"])[:16]
        digest = fingerprint(dict(jobs=jobs, overrides=overrides))
    all_jobs = jobs
    if args.job:
        jobs = [j for j in jobs if fnmatch.fnmatch(j["id"], args.job) or fnmatch.fnmatch(j["label"], args.job)]
    if args.max_jobs is not None:
        if args.max_jobs < 1:
            raise ValueError("max-jobs must be positive")
        jobs = jobs[:args.max_jobs]
    if not jobs:
        raise ValueError("No jobs selected")
    if args.command == "doctor" or args.command == "run" and args.dry_run:
        plan = dict(config=str(args.config), protocol_sha256=digest, jobs_total=len(all_jobs), jobs_selected=len(jobs),
            output=str(output), inference_executed=False, writes=False,
            dependencies={m: importlib.util.find_spec(m) is not None for m in ["numpy", "scipy", "torch", "torchvision", "sklearn"]},
            jobs=[dict(id=j["id"], label=j["label"], phase=j["config"]["phase"],
                estimate=estimate(j["config"]), inputs=inputs(j["config"], overrides)) for j in jobs])
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    version = source_version()
    input_manifest = {j["id"]: inputs(j["config"], overrides, hash_files=True) for j in all_jobs}
    if args.command == "lock":
        if any(j["config"]["phase"] != "confirmation" for j in all_jobs):
            raise ValueError("Lock a confirmation configuration")
        missing = [p["path"] for paths in input_manifest.values() for p in paths if not p["exists"]]
        if missing:
            raise FileNotFoundError("Cannot lock missing inputs: " + ", ".join(sorted(set(missing))))
        development = None
        if args.development_root:
            droot = safe_path(Path(args.development_root) if Path(args.development_root).is_absolute() else REPO / args.development_root)
            development = {str(p.relative_to(droot)): json.loads(p.read_text()) for p in sorted(droot.rglob("DONE.json"))}
            if not development:
                raise ValueError("No completed development jobs in development-root")
        path = output / "protocols" / (digest + ".json")
        write_json(path, dict(protocol_sha256=digest, code_sha256=version["source_sha256"],
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            runtime_dependencies=version.get("dependencies"),
            input_manifest=input_manifest, development_snapshot=development, jobs=all_jobs,
            interpretation="Pre-fixed analysis on held-out search/distribution seeds; old model checkpoints are fixed, not new model trainings."), exclusive=True)
        print(path)
        return 0
    if any(j["config"]["phase"] == "confirmation" for j in jobs):
        if not args.lock:
            raise ValueError("Confirmation requires --lock; first use the lock command after development")
        lock_path = Path(args.lock)
        lock = json.loads((lock_path if lock_path.is_absolute() else REPO / lock_path).read_text())
        if (lock["protocol_sha256"] != digest or lock["code_sha256"] != version["source_sha256"]
                or lock.get("runtime_dependencies") != version.get("dependencies") or lock["input_manifest"] != input_manifest):
            raise ValueError("Confirmation config, code, device or inputs differ from lock")
    failed = []
    for i, job in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {job['id']}", flush=True)
        try:
            print(run_job(job, output, version, overrides, args.resume), flush=True)
        except Exception as exc:
            failed.append(dict(id=job["id"], error=f"{type(exc).__name__}: {exc}"))
            print(failed[-1]["error"], file=sys.stderr, flush=True)
            if not args.keep_going:
                return 1
    if failed:
        print(json.dumps(failed, ensure_ascii=False, indent=2), file=sys.stderr)
    return int(bool(failed))


if __name__ == "__main__":
    raise SystemExit(main())
