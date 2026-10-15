"""Sequential block launcher; every job remains independently resumable."""
from __future__ import annotations

import argparse
import subprocess
import sys
from .common.storage import REPO


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", required=True, help="Explicit configuration files; no implicit full sweep")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-jobs", type=int, help="Limit per config")
    args, forwarded = p.parse_known_args()
    for config in args.configs:
        cmd = [sys.executable, "-m", "research_experiments.run", "run", "--config", config]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.resume:
            cmd.append("--resume")
        if args.max_jobs is not None:
            cmd += ["--max-jobs", str(args.max_jobs)]
        subprocess.run(cmd + forwarded, cwd=REPO, check=True)


if __name__ == "__main__":
    main()
