"""Import existing MNIST arrays.pt initial tensors; no network execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
from .common.storage import REPO, safe_path, write_arrays, write_json, array_hash, file_hash


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("arrays", help="Existing method/arrays.pt from src/mnist_experiments/run.py")
    p.add_argument("--summary", help="Defaults to sibling summary.json")
    p.add_argument("--output", required=True, help="NEW .npz path under research_experiments/results")
    args = p.parse_args(argv)
    source = Path(args.arrays).expanduser().resolve()
    summary = Path(args.summary).expanduser().resolve() if args.summary else source.with_name("summary.json")
    output = safe_path(Path(args.output) if Path(args.output).is_absolute() else REPO / args.output)
    if output.suffix != ".npz" or output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Use a new .npz output path")
    row = json.loads(summary.read_text())
    if not isinstance(row, dict) or row.get("initialization") != "ddim" or not row.get("checkpoint_sha256"):
        raise ValueError("Need a per-method DDIM summary with checkpoint provenance")
    import torch
    data = torch.load(source, map_location="cpu", weights_only=True)
    if not data.get("initial_noise_used", False):
        raise ValueError("The saved run did not use DDIM noise initialization")
    a = data["initial"].detach().cpu().numpy().reshape(len(data["initial"]), -1)
    nfe = row["initialization_nfe"]
    steps = nfe / len(a)
    if int(steps) != steps or steps < 1 or not np.isfinite(a).all():
        raise ValueError("Invalid initialization NFE or array")
    write_arrays(output, initial=a)
    write_json(output.with_suffix(".json"), dict(format="dms_saved_starts_v1", seed=row["initial_noise_seed"],
        ddim_steps=int(steps), initialization_nfe=nfe, checkpoint_sha256=row["checkpoint_sha256"], array_hash=array_hash(a),
        original_arrays_sha256=file_hash(source), original_summary_sha256=file_hash(summary),
        interpretation="Historical starts reused, not newly independent confirmation"))
    print(output)


if __name__ == "__main__":
    main()
