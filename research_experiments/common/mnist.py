"""Frozen evaluator and existing kNN definitions; everything saved for analysis."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from .config import input_path
from .storage import file_hash, fingerprint, array_hash, write_arrays, write_json


def evaluate_mnist(result, field, config, output, overrides):
    import torch
    from src.mnist_experiments.evaluator import FrozenEvaluator
    from src.mnist_experiments.data import ReferenceData
    from src.mnist_experiments.evaluation import nearest_neighbors, support_percentiles, semantic_metrics
    cfg = config["mnist"]
    evaluator_path = input_path(cfg["evaluator_checkpoint"], overrides)
    if not evaluator_path.is_file():
        raise FileNotFoundError(f"Independent evaluator missing: {evaluator_path}. No training will run.")
    evaluator = FrozenEvaluator.load(evaluator_path, device=field.device)
    data = ReferenceData(field.state, root=str(input_path(cfg["data_root"], overrides)),
                         download=False, include_test=cfg["include_test"])
    # Assert evaluator and diffusion train/validation partitions are identical.
    for split in ["train", "validation"]:
        recorded = evaluator.metadata.get(split + "_indices")
        if recorded is None or not torch.equal(torch.as_tensor(recorded).long().cpu(), data.indices[split]):
            raise ValueError("Evaluator and diffusion splits differ")
    train, validation = data.images("train", cfg["train_limit"]), data.images("validation", cfg["validation_limit"])
    key = fingerprint(dict(evaluator=file_hash(evaluator_path), train_indices=data.indices["train"][:cfg["train_limit"]].tolist(),
        validation_indices=data.indices["validation"][:cfg["validation_limit"]].tolist(), k=cfg["knn_k"],
        pixel_hash=array_hash(train.numpy()), torch=str(torch.__version__), device=field.device))
    cache = Path(output) / "reference_cache" / (key + ".npz")
    if cache.exists():
        with np.load(cache, allow_pickle=False) as z:
            train_features = torch.from_numpy(z["train_features"])
            refs = {name: torch.from_numpy(z[name]) for name in ["pixel", "feature"]}
    else:
        train_features, val_features = evaluator.features(train), evaluator.features(validation)
        refs = {}
        for name, q, r in [("pixel", validation, train), ("feature", val_features, train_features)]:
            dd, _ = nearest_neighbors(q, r, cfg["knn_k"])
            refs[name] = dd.mean(1) / (math.sqrt(784) if name == "pixel" else 1.)
        write_arrays(cache, train_features=train_features.numpy(), **{k: v.numpy() for k, v in refs.items()})
    second = None
    if cfg["second_evaluator_checkpoint"]:
        second = FrozenEvaluator.load(input_path(cfg["second_evaluator_checkpoint"], overrides), device=field.device)
    stages = [s for s in result["snapshots"] if
        s["stage"] in {"initial", "before_final", "after_final", "output"} or
        s["stage"].startswith("after_level_")]
    if not cfg["stages"]:
        stages = [stages[0], stages[-1]]
    initial = torch.as_tensor(result["initial"][:config["population"]], dtype=torch.float32).reshape(-1, 1, 28, 28)
    initial_logits, _ = evaluator.evaluate(initial)
    initial_labels = initial_logits.argmax(1)
    selected_ids = [i for i in cfg["diagnostic_indices"] if i < len(initial)]
    arrays, records, previous_neighbors = {}, [], {}
    for index, stage in enumerate(stages):
        x = torch.as_tensor(stage["x"], dtype=torch.float32).reshape(-1, 1, 28, 28)
        if not len(x):
            records.append(dict(stage=stage["stage"], n=0))
            continue
        logits, features = evaluator.evaluate(x)
        sem = semantic_metrics(logits)
        prefix = f"stage{index:03d}_"
        arrays[prefix + "images"] = x.numpy()
        arrays[prefix + "features"] = features.numpy()
        arrays[prefix + "logits"] = logits.numpy()
        row = dict(stage=stage["stage"], index=index, t=stage["t"], n=len(x),
            class_histogram=sem["class_histogram"].tolist(), class_entropy=sem["class_entropy"],
            class_coverage=sem["class_coverage"], mean_confidence=sem["mean_confidence"],
            initial_confidence=initial_logits.softmax(1).max(1).values.tolist(), lineage=stage["lineage"])
        lineage = stage["lineage"]
        if lineage is not None:
            # A merged representative has several parents; save all comparisons.
            transitions = [dict(candidate=j, parents=ids,
                fraction_changed=float((initial_labels[ids] != sem["predicted_class"][j]).float().mean()))
                for j, ids in enumerate(lineage) if ids and max(ids) < len(initial)]
            row["transitions"] = transitions
            row["class_change_fraction"] = float(np.mean([t["fraction_changed"] for t in transitions])) if transitions else None
        elif len(x) == len(initial) and config["method"]["merge"]["when"] == "none" and config["method"]["selection"]["kind"] == "none":
            row["class_change_fraction"] = float((initial_labels != sem["predicted_class"]).float().mean())
        for name, query, ref in [("pixel", x, train), ("feature", features, train_features)]:
            distance, idx = nearest_neighbors(query, ref, cfg["knn_k"])
            distance /= math.sqrt(784) if name == "pixel" else 1.
            knn = distance.mean(1)
            arrays[prefix + name + "_knn"] = knn.numpy()
            arrays[prefix + name + "_neighbors"] = data.indices["train"][idx].numpy()
            row[name + "_knn"] = float(knn.mean())
            row[name + "_support"] = float(np.mean(support_percentiles(knn, refs[name])["support_percentile"]))
            nearest = data.indices["train"][idx[:, 0]]
            if name in previous_neighbors and len(previous_neighbors[name]) == len(nearest) and config["method"]["merge"]["when"] == "none":
                row[name + "_nearest_change_fraction"] = float((nearest != previous_neighbors[name]).float().mean())
            previous_neighbors[name] = nearest
        flat = x.flatten(1)
        if len(x) > 1:
            dist = torch.cdist(flat, flat, compute_mode="donot_use_mm_for_euclid_dist") / 28
            ii = torch.triu_indices(len(x), len(x), 1)
            pairs = dist[ii[0], ii[1]]
            same = sem["predicted_class"][ii[0]] == sem["predicted_class"][ii[1]]
            row.update(pairwise_rms=float(pairs.mean()), near_duplicate_fraction=float((pairs < .02).float().mean()),
                within_class_rms=float(pairs[same].mean()) if same.any() else None)
        if cfg["include_test"]:
            td, _ = nearest_neighbors(x, data.images("test"), 1)
            row["nearest_test_rms"] = float(td.mean() / 28)
            arrays[prefix + "nearest_test"] = (td[:, 0] / 28).numpy()
        if second:
            sl, _ = second.evaluate(x)
            arrays[prefix + "second_logits"] = sl.numpy()
            row["second_classifier_agreement"] = float((sl.argmax(1) == logits.argmax(1)).float().mean())
        records.append(row)
    return dict(stages=records, reference_key=key, train_count=len(train), validation_count=len(validation),
        diagnostic_indices=selected_ids, evaluator_sha256=file_hash(evaluator_path),
        evaluator_images=evaluator.images_evaluated, evaluator_calls=evaluator.forward_calls,
        second_evaluator_images=second.images_evaluated if second else 0,
        interpretation="Empirical support/semantics; neither classifier classes nor kNN identify true MNIST modes."), arrays
