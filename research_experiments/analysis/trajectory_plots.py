"""Offline movement, alignment, basin and split diagnostics from saved JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from ..common.storage import REPO, safe_path
from .analyze import write_csv


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    job = Path(args.job) if Path(args.job).is_absolute() else REPO / args.job
    output = safe_path(Path(args.output) if Path(args.output).is_absolute() else REPO / args.output)
    if output.exists():
        raise FileExistsError(output)
    if not (job / "DONE.json").exists():
        raise ValueError("Completed job required")
    output.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    movement = json.loads((job / "movement.json").read_text())
    rows = []
    for r in movement:
        row = {k: v for k, v in r.items() if not isinstance(v, list)}
        for k, v in r.items():
            if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
                row[k + "_mean"] = float(np.mean(v))
        rows.append(row)
    write_csv(output / "movement.csv", rows)
    alignment = json.loads((job / "alignment.json").read_text()) if (job / "alignment.json").exists() else []
    flattened = []
    for r in alignment:
        row = {k: v for k, v in r.items() if not isinstance(v, list)}
        for k in ["exact_norm", "learned_norm", "cosine", "relative_norm_error", "distance_to_reference", "basin_residual"]:
            if k in r:
                row[k + "_mean"] = float(np.mean(r[k]))
        if "exact_clean_basin" in r:
            row["unverified_basin_fraction"] = float(np.mean(np.asarray(r["exact_clean_basin"]) < 0))
        flattened.append(row)
    write_csv(output / "alignment.csv", flattened)
    if (job / "split_diagnostics.json").exists():
        write_csv(output / "split_probes.csv", json.loads((job / "split_diagnostics.json").read_text()))
    for label, table, metrics in [("movement", rows, ["score_norm_mean", "actual_l2_mean", "proposed_l2_mean", "displacement_mean", "reversal_fraction", "outside_fraction"]),
            ("alignment", flattened, ["cosine_mean", "relative_norm_error_mean", "distance_to_reference_mean", "unverified_basin_fraction"])]:
        for metric in metrics:
            good = [(i, r[metric]) for i, r in enumerate(table) if isinstance(r.get(metric), (int, float))]
            if not good:
                continue
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot([i for i, _ in good], [v for _, v in good])
            ax.set(xlabel="saved update/stage", ylabel=metric)
            fig.tight_layout()
            fig.savefig(output / (label + "_" + metric + ".png"), dpi=160)
            plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
