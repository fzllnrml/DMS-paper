# PROJECT_MAP — preliminary navigation guide

This is a **file-structure-derived index**, not a verified code review.
Read the actual source and configurations before claiming any functionality.

- `src/algorithms/`: implementations of mode-finding/search algorithms.
- `src/mnist/`: MNIST diffusion/data/train/sample modules.
- `src/mnist_experiments/`: MNIST experiment execution, evaluation, search, aggregation.
- `src/mnist_controls/`: control/confirmation experiments.
- `src/mnist_dimension/`: dimension studies.
- `src/models/`, `src/utils/`: common model and utility code.
- `configs/`: reproducible experiment configurations.
- `scripts/`: runnable experiment scripts and utilities.
- `tests/`: tests. Review all dependencies before running.
- `levine/`: Levine 13D-related code (data/results optional).
- `runs/**/checkpoints/`: trained models copied with original relative paths.
- `paper_results_archive/`: curated summaries and original config metadata.
- `results_pareto_validation_all/`: top-level comparative summaries only.
- `CHECKPOINT_MAP.csv`: SHA-256 and canonical weight path for each model file.
- `EXPERIMENT_INDEX.md`: approximate catalog of result groups.

## Instructions for a new coding agent

1. Read `README.md`, `MNIST_RUNBOOK.md`, `MNIST_EXPERIMENTS.md`, and this map.
2. Identify relevant actual code and experiment configs; do not load every result file.
3. Inspect `CHECKPOINT_MAP.csv` before proposing model retraining.
4. Existing model seeds 42, 43, 44, plus evaluator are kept by path where present.
5. Source project's full raw outputs are **not** part of this checkout. Do not
   claim they are missing scientifically; they remain in the original archive.
6. Write experiment code and reproducible CLI invocations. The human will run
   compute-intensive experiments locally; do not retrain or launch sweeps.
7. Compare length-of-step controls vs search, and keep the research hypothesis
   separate from already measured results.
