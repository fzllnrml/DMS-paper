"""Independent frozen MNIST evaluation, with V2-only reference caches.

Same evaluator, pixel RMS, feature L2 and direct-distance kNN as the repository.
New diagnostics compare training proximity with held-out validation proximity;
they detect suspicious closeness, not prove or disprove memorization.
"""
from __future__ import annotations

import numpy as np

from .storage import RESULTS, input_path, file_hash, array_hash, fingerprint, write_arrays, runtime_identity


class MNISTEvaluator:
    def __init__(self, field, config, overrides=None):
        import torch
        from src.mnist_experiments.evaluator import FrozenEvaluator
        from src.mnist_experiments.data import ReferenceData
        from src.mnist_experiments.evaluation import nearest_neighbors
        self.cfg = config["mnist"]
        cfg = self.cfg
        p = input_path(cfg["evaluator_checkpoint"], overrides)
        self.evaluator = FrozenEvaluator.load(p, device=field.device)
        self.data = ReferenceData(field.state, root=str(input_path(cfg["data_root"], overrides)), download=False)
        for split in ["train", "validation"]:
            recorded = self.evaluator.metadata.get(split + "_indices")
            if recorded is None or not torch.equal(torch.as_tensor(recorded).long().cpu(), self.data.indices[split]):
                raise ValueError("Diffusion and frozen evaluator train/validation splits differ")
        self.train = self.data.images("train", cfg["train_limit"])
        self.validation = self.data.images("validation", cfg["validation_limit"])
        identity = dict(evaluator_sha256=file_hash(p), train_sha256=array_hash(self.train.numpy()),
                        validation_sha256=array_hash(self.validation.numpy()), k=cfg["knn_k"],
                        train_indices=self.data.indices["train"][:len(self.train)].tolist(),
                        validation_indices=self.data.indices["validation"][:len(self.validation)].tolist(),
                        runtime=runtime_identity(), device=field.device)
        self.key = fingerprint(identity)
        cache = RESULTS / "cache" / "mnist_references" / (self.key + ".npz")
        if cache.is_file():
            with np.load(cache, allow_pickle=False) as z:
                self.train_features = torch.from_numpy(z["train_features"])
                self.validation_features = torch.from_numpy(z["validation_features"])
                self.calibration = {k: z[k].copy() for k in ["pixel_nearest", "pixel_knn", "feature_nearest", "feature_knn"]}
        else:
            self.train_features = self.evaluator.features(self.train)
            self.validation_features = self.evaluator.features(self.validation)
            self.calibration = {}
            for name, q, r, scale in [("pixel", self.validation, self.train, 28.),
                                       ("feature", self.validation_features, self.train_features, 1.)]:
                distances, _ = nearest_neighbors(q, r, cfg["knn_k"])
                self.calibration[name + "_nearest"] = distances[:, 0].numpy() / scale
                self.calibration[name + "_knn"] = distances.mean(1).numpy() / scale
            write_arrays(cache, train_features=self.train_features.numpy(),
                         validation_features=self.validation_features.numpy(), **self.calibration)
        self.identity = dict(identity, reference_key=self.key, train_count=len(self.train),
                             validation_count=len(self.validation), test_used=False,
                             distance_units=dict(pixel="RMS normalized pixels", feature="L2 frozen features"))
        self._cache = {}

    def inspect(self, x, support=True):
        import torch
        from src.mnist_experiments.evaluation import nearest_neighbors, semantic_metrics, support_percentiles
        x = np.asarray(x)
        key = (array_hash(x), support)
        if key in self._cache:
            return self._cache[key]
        if not len(x):
            return dict(n_candidates=0, class_coverage=0), {}
        images = torch.as_tensor(x, dtype=torch.float32).reshape(-1, 1, 28, 28)
        logits, features = self.evaluator.evaluate(images)
        sem = semantic_metrics(logits)
        labels = sem["predicted_class"]
        arrays = dict(images=images.numpy(), logits=logits.numpy(), features=features.numpy(), predicted_class=labels.numpy())
        row = {k: sem[k] for k in ["class_coverage", "class_entropy", "mean_confidence"]}
        row.update(n_candidates=len(x), class_histogram=sem["class_histogram"].tolist())
        if len(images) > 1:
            dist = torch.cdist(images.flatten(1), images.flatten(1), compute_mode="donot_use_mm_for_euclid_dist") / 28
            ii = torch.triu_indices(len(images), len(images), 1)
            pairs = dist[ii[0], ii[1]]
            same = labels[ii[0]] == labels[ii[1]]
            row.update(pairwise_rms=float(pairs.mean()),
                       near_duplicate_fraction=float((pairs < self.cfg["near_duplicate_rms"]).float().mean()),
                       within_class_rms=float(pairs[same].mean()) if same.any() else None)
        else:
            row.update(pairwise_rms=None, near_duplicate_fraction=None, within_class_rms=None)
        if support:
            for name, query, train, validation, scale in [
                ("pixel", images, self.train, self.validation, 28.),
                ("feature", features, self.train_features, self.validation_features, 1.)]:
                distances, idx = nearest_neighbors(query, train, self.cfg["knn_k"])
                vd, _ = nearest_neighbors(query, validation, self.cfg["knn_k"])
                distances, vd = distances.numpy() / scale, vd.numpy() / scale
                nearest, knn = distances[:, 0], distances.mean(1)
                threshold = float(np.quantile(self.calibration[name + "_nearest"], self.cfg["train_close_quantile"]))
                row.update({name + "_knn": float(knn.mean()), name + "_nearest": float(nearest.mean()),
                            name + "_validation_knn": float(vd.mean()),
                            name + "_support": float(np.mean(support_percentiles(knn, self.calibration[name + "_knn"])["support_percentile"])),
                            name + "_train_extreme_close_fraction": float(np.mean(nearest < threshold)),
                            name + "_train_close_threshold": threshold,
                            name + "_train_validation_nearest_ratio": float(np.mean(nearest / np.maximum(vd[:, 0], 1e-12))),
                            name + "_exact_train_copy_fraction": float(np.mean(nearest < 1e-6))})
                arrays[name + "_neighbor_indices"] = self.data.indices["train"][idx].numpy()
                arrays[name + "_neighbor_distances"] = distances
                arrays[name + "_validation_distances"] = vd
                arrays[name + "_train_close"] = nearest < threshold
        self._cache[key] = (row, arrays)
        return row, arrays

    def evaluate(self, result, outputs):
        # Clear candidate cache between jobs. Reference features stay in memory.
        self._cache.clear()
        before_images, before_calls = self.evaluator.images_evaluated, self.evaluator.forward_calls
        initial_row, initial_arrays = self.inspect(result["initial"])
        initial_labels = initial_arrays["predicted_class"]
        arrays, rows, stages = {}, {}, []
        for stage in result["snapshots"]:
            if stage["stage"] == "initial" or stage["stage"].startswith("after_level_") or stage["stage"] == "after_final":
                row, a = self.inspect(stage["x"], support=stage["stage"] == "initial")
                stages.append(dict(row, stage=stage["stage"], t=stage["t"], lineage=stage["lineage"]))
                prefix = f"stage{len(stages)-1:03d}_"
                arrays.update({prefix + k: v for k, v in a.items()})
        for name, out in outputs.items():
            row, a = self.inspect(out["candidates"])
            row = dict(row)
            ancestry = out["lineage"]
            if ancestry is not None and len(ancestry):
                changes, distances = [], []
                for j, parents in enumerate(ancestry):
                    changes.append(float(np.mean(initial_labels[parents] != a["predicted_class"][j])))
                    distances.append(float(np.mean(np.linalg.norm(result["initial"][parents] - out["candidates"][j], axis=1) / 28)))
                row.update(class_change_fraction=float(np.mean(changes)), displacement_from_ddim_rms=float(np.mean(distances)))
                a = dict(a, per_candidate_class_change=np.array(changes), per_candidate_displacement_rms=np.array(distances))
            rows[name] = row
            arrays.update({"output_" + name + "_" + k: v for k, v in a.items()})
        report = dict(reference=self.identity, initial=initial_row, stages=stages, outputs=rows,
                      evaluator_images=self.evaluator.images_evaluated-before_images,
                      evaluator_forward_calls=self.evaluator.forward_calls-before_calls,
                      evaluator_cost_is_outside_search_nfe=True,
                      interpretation="Empirical support and classifier semantics, not true MNIST modes; proximity flags are diagnostics, not a memorization proof")
        return report, arrays
