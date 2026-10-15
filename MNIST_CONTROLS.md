# Frozen MNIST pilot controls

Apply these added files at the root of the existing repository. Existing publication
modules, definitions, outputs, and training recipes are unchanged. The source model
is `src/mnist/model.py` (there is no `src/mnist_experiments/model.py`).

## Environment and inputs

Use your existing MNIST environment. If needed:

```bash
source .venv-mnist/bin/activate
python -m pip install -r requirements-mnist-experiments.txt
```

Default required inputs:

- `runs/mnist/checkpoints/best.pt`, model seed42, original T=1000 pixel diffusion;
- `runs/mnist_evaluator/best.pt`, same train/validation split;
- downloaded MNIST at `data/mnist/MNIST/raw/`;
- existing publication reference caches at
  `runs/mnist_experiments/publication_reference/` (same PublicationSupport class).

Paths can be set in `configs/mnist_controls.yaml`. Scientific parameters and evaluation
settings are frozen. No implicit training or dataset download is performed. Run from
the repository root. CPU, CUDA and MPS use the original model's device support.

Old publication tensors are optional. The runner discovers seed42/P128 suites under
`publication_root`, or accepts explicit `publication_suites: {0: path, 1: path, 2: path}`.
A reusable bundle must contain metadata/config plus complete matched-DDIM and
RMS-search arrays/summary/completion files. Complete bundles are checked for original
implementation/definition hashes, weights, evaluator, starts, prefix pairing and
completion consistency. Incompatible complete bundles fail visibly. Missing/incomplete
bundles trigger regeneration using the unchanged publication definitions, not aggregate
numbers. Set publication_root to a nonexistent path and publication_suites to `{}`
before starting a distinct run if deliberate fresh regeneration is required.

Imported baseline images are evaluated with the exact same PublicationSupport instance
as the new methods. Old diagnostic values are never reused as stability evidence.
The reference cache is reused by its original content-derived key, and pinned by
checksum for resume. Classifier evaluation and reference-cache construction cost are
separate from diffusion NFE. The original exact kNN evaluator can be a substantial
CPU cost; NFE is not a wall-time estimate.

## Commands

```bash
# Metadata-only plan: no checkpoint loading, inference, training or dataset download.
python -m src.mnist_controls.runner --config configs/mnist_controls.yaml --dry-run

# Actual pilot on Apple Silicon. This was NOT executed during implementation.
python -m src.mnist_controls.runner --config configs/mnist_controls.yaml --execute --device mps

# Same configuration/device/inputs, completed jobs skipped.
python -m src.mnist_controls.runner --config configs/mnist_controls.yaml --execute --resume --device mps

# Rebuild tables; returns nonzero if expected jobs are missing/failed/corrupt.
python -m src.mnist_controls.runner --config configs/mnist_controls.yaml --aggregate

# Cheap unit + integration smoke: fake MNIST, tiny random real U-Net, no training/download.
python -m unittest discover -s tests -p 'test_mnist_controls.py' -v
```

The optional `--phase pilot` is accepted. No confirmation launcher or hyperparameter
sweep exists. With no execution flag, the default operation is dry-run.

## Work and accounting

Pilot: model42, seeds0/1/2, P128. Exactly **9 quality jobs + 6 stability jobs**.
Three additional shared-input dependency jobs import or regenerate paired inputs;
they are identified separately in the inventory.

| Item | Standalone NFE per seed | New NFE per seed with shared inputs ready |
|---|---:|---:|
| DDIM-105 | 13440 | 13440 |
| Low-noise-only | 13440 | 7040 |
| Generate-and-select | 13400 | 0 |
| Stability of RMS + DDIM | 12800 diagnostic | 12800 |
| Total | 53080 | 33280 |

Across three seeds: **159240** individually attributed NFE, **99840** new NFE with old
paired input bundles, at most **161160** physically executed NFE if all shared inputs
must be regenerated. Regeneration per seed costs13400 for the shared pool plus up to7040
for full publication search. Its original early stopping is preserved; actual costs
are recorded. New low-noise and stability maps always perform their fixed step counts.
Repeated calls lost before completion after an interruption are not included in
completed-artifact accounting; failed attempts are retained in status/runtime records.

DDIM-105 uses the original deterministic sampler, 105 unique rounded linspace indices
999→0. Duplicate indices cause an explicit incompatibility error rather than silently
changing its105-evaluation budget. Low-noise uses45 steps .03, then10 .003, all t=0,
RMS geometry and clamp. Selector uses only generated pixels, CPU float64 RMS distances,
fixed deterministic greedy medoids,128 original indices. Its wall time, matrix bytes,
working-array estimate and operation counts are separate; no evaluator enters selection.

Stability uses two directions and scales .02/.05, actual post-clipping delta0>.01,
no resampling, and the same ten-step reference/perturbed map. Ratios for invalid
perturbations are JSON null, not fabricated zero contraction. All-invalid groups remain
visible. Median uses the conventional midpoint median. This measures contraction of a
specific projected map, not a mathematical mode.

## Files and resume

Outputs: `runs/mnist_controls/model42_seed{0,1,2}/`.
Each method and `inputs/` has resolved config, arrays, metrics, provenance,
summary, runtime/NFE, status and checksummed completion marker.

Summary directory `runs/mnist_controls/summary/` contains:

- `raw_runs.csv/json`: all quality/baseline rows, including failed/missing placeholders;
- `summary.csv/json`: expected/complete counts and observed mean/std;
- `paired_comparisons.csv/json`: each paired seed, means, practical margins, gain ratios;
- `stability_v2.csv/json` and `stability_v2_per_candidate.csv/json`;
- `job_status.csv/json`:15 pilot jobs plus3 shared-input dependencies;
- `decisions.json`, `compute.json`, `controls_summary.md`;
- `quality_diversity.png/pdf` using only existing coverage, entropy, feature kNN.

There are18 quality/baseline raw rows in a complete pilot (six methods × three seeds),
not18 newly generated quality-control jobs. Pairing of the268→128 selector is explicitly
set-level; there is no artificial candidate-by-candidate pairing with128 search outputs.

The output lock prevents concurrent writers. Resume pins source code, configuration,
weights, evaluator, raw dataset files, old tensor bundle and reference cache hashes.
A valid completed job is never overwritten. Corrupt/incompatible completed jobs fail
closed; do not repair them by silently changing parameters. Failed/partial jobs retry
with the same seed. Independent jobs/seeds continue after a method failure. Tables
retain failures and expected seeds, and never report partial observations as a complete
pilot. Any automatic pilot decision includes raw values and does not launch more work.

Estimated additional disk: approximately **60–100 MB** for pilot tensors/tables/images
and a newly needed reference cache; reserve **200 MB**, excluding already installed
MNIST and model/evaluator checkpoints. No model training or weight copies are created.
The selector's distance matrix alone is268²×8=574592 bytes; neural model memory is separate.

## Scope and validation

Tests cover genuine sampler calls/NFE, zero-score no-early-stop refinement, pairing,
selector membership/API/ties, identity/contraction/translation maps, clipping/invalid
ratios, complete/partial resume, cache/checkpoint corruption, failed-job visibility,
all three seeds, and exact publication evaluator/cache reuse. Integration uses a random
small real U-Net and synthetic MNIST data, not a trained scientific model.

The source snapshot's original files are checked byte-for-byte after implementation.
No old publication definitions or results are edited. The fixed scientific plan remains
in `NEXT_EXPERIMENTS.md`; this implementation adds only its requested pilot stage.

Implementation verification (CPU, 2026-10-13): **42 passed, 2 skipped** across
`test_mnist_controls.py` (15 tests), `test_mnist_experiments.py`, and
`test_mnist_publication.py`. CUDA/MPS backend tests were skipped because that hardware
is unavailable here. The uploaded repository snapshot omits `levine/`; for the old
Levine regression test only, that package was read from the original GitHub repository
via PYTHONPATH. No missing package or test was replaced by a mock. Dry-run and compile
checks passed. All pre-existing snapshot files remained byte-identical. No real pilot
or scientific training was executed.
