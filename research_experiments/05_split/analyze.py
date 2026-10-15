"""Offline common analysis for this block."""
from pathlib import Path
import sys


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from research_experiments.analysis.analyze import main as analyze
    return analyze()


if __name__ == "__main__":
    main()
