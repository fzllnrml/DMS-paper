"""Render fixed-index image trajectories and semantic/kNN curves offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from ..common.storage import REPO, safe_path, write_json


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("job", help="Completed MNIST job directory")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    job = Path(args.job) if Path(args.job).is_absolute() else REPO / args.job
    output = safe_path(Path(args.output) if Path(args.output).is_absolute() else REPO / args.output)
    if output.exists():
        raise FileExistsError(output)
    if not (job / "DONE.json").exists():
        raise ValueError("Only completed saved jobs can be plotted")
    meta = json.loads((job / "mnist.json").read_text())
    cfg = json.loads((job / "resolved_config.json").read_text())
    stages = meta["stages"]
    ids = meta["diagnostic_indices"]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output.mkdir(parents=True)
    with np.load(job / "mnist.npz", allow_pickle=False) as data:
        fig, axes = plt.subplots(len(ids), len(stages), figsize=(1.5 * len(stages), 1.5 * len(ids)), squeeze=False)
        for col, stage in enumerate(stages):
            images = data[f"stage{stage['index']:03d}_images"]
            logits = data[f"stage{stage['index']:03d}_logits"]
            for row, original_id in enumerate(ids):
                ax = axes[row, col]
                lineage = stage.get("lineage")
                if lineage is not None:
                    matches = [i for i, group in enumerate(lineage) if original_id in group]
                elif len(images) == cfg["population"] and cfg["method"]["merge"]["when"] == "none" and cfg["method"]["selection"]["kind"] == "none":
                    matches = [original_id]
                else:
                    matches = []
                if matches:
                    i = matches[0]
                    ax.imshow(images[i, 0], cmap="gray", vmin=-1, vmax=1)
                    ax.set_xlabel(f"initial {original_id}; class {int(logits[i].argmax())}", fontsize=6)
                else:
                    ax.text(.5, .5, "no unique ancestry", ha="center", fontsize=6)
                ax.set_xticks([])
                ax.set_yticks([])
                if row == 0:
                    ax.set_title(stage["stage"], fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "fixed_trajectories.png", dpi=160)
        plt.close(fig)
    for metric in ["pixel_knn", "feature_knn", "class_change_fraction", "class_entropy", "near_duplicate_fraction"]:
        pairs = [(s["stage"], s[metric]) for s in stages if metric in s]
        if not pairs:
            continue
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.plot(range(len(pairs)), [v for _, v in pairs], marker="o")
        ax.set_xticks(range(len(pairs)), [name for name, _ in pairs], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(metric)
        fig.tight_layout()
        fig.savefig(output / (metric + ".png"), dpi=160)
        plt.close(fig)
    write_json(output / "selection_rule.json", dict(indices=ids, rule="Preconfigured initial indices, every available stage; no quality-based selection"))
    print(output)


if __name__ == "__main__":
    main()
