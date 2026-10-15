"""Run this independent block; no execution on import."""
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research_experiments.run import main as cli
    args = list(sys.argv[1:])
    phase = "confirmation" if "--confirmation" in args else "pilot"
    if "--confirmation" in args:
        args.remove("--confirmation")
    return cli(["run", "--config", f"research_experiments/configs/01_step_size.{phase}.json", *args])


if __name__ == "__main__":
    raise SystemExit(main())
