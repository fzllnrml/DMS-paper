# Frozen MNIST confirmation

This patch extends the existing controls engine to model seeds 43/44 and search
seeds 0–4. It does not change `controls.py`, any publication definition, the pilot
YAML, or any pilot result. No training is performed. All three controls and both
stability diagnostics use exactly the pilot implementations and parameters.

From the repository root, use the existing `.venv-mnist` environment:

```bash
source .venv-mnist/bin/activate
python -m pytest -q tests/test_mnist_controls.py tests/test_mnist_controls_confirmation.py tests/test_mnist_publication.py
python -m src.mnist_controls.confirmation --config configs/mnist_controls_confirmation.yaml --dry-run
python -m src.mnist_controls.confirmation --config configs/mnist_controls_confirmation.yaml --execute --device mps
python -m src.mnist_controls.confirmation --config configs/mnist_controls_confirmation.yaml --execute --resume --device mps
python -m src.mnist_controls.confirmation --config configs/mnist_controls_confirmation.yaml --aggregate
```

Only execute after reviewing your local dry-run. Dependencies are unchanged
(`requirements-mnist-experiments.txt`); there is no dataset download or implicit
model training. Dry-run inspects metadata/file presence, not tensors or models.
Missing checkpoints/evaluator/raw MNIST files are listed explicitly.

## Inputs and pairing

The separate YAML defaults to the original publication paths:

- model43: `runs/mnist_seed43/checkpoints/best.pt`
- model44: `runs/mnist_seed44/checkpoints/best.pt`
- evaluator: `runs/mnist_evaluator/best.pt`
- data: `data/mnist`
- old tensors: `runs/mnist_experiments/publication_main`
- reference cache: `runs/mnist_experiments/publication_reference`
- read-only pilot: `runs/mnist_controls`

Only filesystem paths/device may be adjusted. For ambiguous old suites, set
`publication_suites: {43: {0: path_to_suite}, 44: {0: another_suite}}` as needed.
Automatic discovery distinguishes model seed, search seed and P=128.
Old tensors are optional. Complete bundles undergo exactly the pilot checks:
publication definition/code hashes, checkpoint/evaluator hashes, config, seed,
completion markers, Gaussian prefix, DDIM prefix and candidate validity. An
incompatible complete bundle fails; it is not silently replaced by fresh data.
Missing bundles regenerate the common DDIM pool and full search using unchanged
publication definitions. Evaluation uses the same PublicationSupport and caches.

Each model/search seed shares one Gaussian pool, its DDIM-50 prefix, and full
search endpoint. Selection is paired at set level, not pointwise to a different
population count. No pilot outcome is read by generation or selection.

## Inventory and budget

There are **30 quality + 20 diagnostic + 10 preparation jobs**.

| Work | NFE across 10 model/search pairs |
|---|---:|
| DDIM-105, 128 outputs | 134,400 new |
| Low-noise-only movement | 70,400 new; 64,000 reused initialization |
| Pool representatives | 0 new; 134,000 reused pool |
| Stability on full search and DDIM initializations | 128,000 new |
| Preparation with compatible old bundles | 0 new |
| Preparation with no old tensors | at most 204,400 new |

Thus physical new diffusion NFE is **332,800 with all old bundles**, up to
**537,200 without them**. Mixed reuse adds at most 20,440 per missing bundle.
Fair standalone controls+diagnostics sum to **530,800**. Shared reuse is not
extra computation; full-search early exit can reduce regeneration cost. Evaluator
calls, CPU selection and failed/interrupted attempts are separate from those NFE
budgets. Actual completed-artifact work appears in `summary/compute.json`.

Allow approximately **200–350 MB** additional output (tensors, raw diagnostics,
JSON/CSV and combined tables); reserve **1 GB**, excluding existing weights,
MNIST and any absent reference cache. This is an estimate, not a measured heavy run.

## Outputs, aggregation and resume

New artifacts are exclusively under `runs/mnist_controls_confirmation/`:

- root config/plan/manifest and last invocation errors;
- `model43/model43_seed0/...` (similarly all ten pairs): the same atomic job
  artifacts, provenance, status, runtime, NFE and tensor checksums as pilot;
- `summary/`: `raw_runs.csv`, `summary.csv`, `paired_comparisons.csv`,
  `stability_v2.csv`, `stability_v2_per_candidate.csv`, `job_status.csv`, JSON
  counterparts, `decisions.json`, `compute.json`, `controls_summary.md`;
- `summary/combined/`: read-only-source pilot+confirmation tables, `phase` labels
  and source manifest hashes. Existing pilot CSVs are never regenerated/overwritten.

Aggregation verifies job checksums, retains all expected model/search seeds and
failed/missing rows, and never converts incomplete paired samples to a complete
confidence interval. It can run without model weights or inference. Missing pilot
artifacts do not block confirmation execution; combined provenance explicitly
reports `complete: false`. The pilot's evaluator/reference-cache identity must
match before combined analysis is marked complete. Phases stay separate; pilot
is never included in confirmation confidence intervals.

Confirmation inference follows NEXT_EXPERIMENTS.md: model-wise five-seed paired
means, margins .0069/.196, 10,000 stratified bootstrap resamples, RNG 20261013,
equal model weights, six co-primary contrasts with central 99.1667% intervals
(family 95%). The low-noise-tie branch is not activated: this series follows the
reported clear-win pilot. Diversity guardrails and stability criteria are unchanged.
Decisions never hide raw values or launch another series.

`--resume` skips valid completed jobs and retries only incomplete ones with the
same configuration. Config/code/checkpoint/evaluator/data/reference/source hash
changes are rejected. Failed models do not prevent recording the other model's
jobs. Exit code is nonzero for an incomplete confirmation. A new context requires
a separate output root; never delete integrity markers to bypass checks.

The orchestration code hash changes with this patch. Old pilot artifacts remain
readable for combined analysis without migration or rewriting. An attempt to
resume an old pilot with changed code is deliberately rejected by its existing
hash guard; this patch never resumes or reruns model42. Keep the original pilot
patch/environment when exact historical execution replay is needed.

Tests use tiny randomly initialized models, synthetic MNIST and temporary result
fixtures. They are software checks, not confirmation observations.
