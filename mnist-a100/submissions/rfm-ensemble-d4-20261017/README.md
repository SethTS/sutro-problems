# RFM kernel ensemble: difficulty 4

Non-neural RBF kernel ensemble for **difficulty 4**, with **three analytic metric updates per model**, Nyström-preconditioned conjugate-gradient solves and local affine-hull correction. All three initial energy-enabled sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`) passed: **218.463697 ms/call median**, **39,296.268 mJ/call median above idle**, and **2.209697% pooled MNIST error** across 33 MNIST calls.

Difficulty 4 requires mean MNIST error at most **2.30% within each run**, individual-draw error at most **3.80%**, and holdout accuracy at least **15%**.

| Run | GPU | Ranked ms/call | Above-idle mJ/call | MNIST error | Worst draw | Holdout accuracy |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | A100-SXM4-80GB | 217.963516 | 39,296.268 | 2.216364% | 2.49% | Fashion 88.6300% |
| 2 | A100-80GB PCIe | 219.338938 | 39,088.839 | 2.242727% | 2.57% | KMNIST 96.9575% |
| 3 | A100-SXM4-80GB | 218.463697 | 43,198.217 | 2.170000% | 2.50% | Fashion 88.2675% |

## Method and runtime

Five classifiers use different cyclic orders of the labelled training rows. Each learns an RBF feature metric through three analytic average-gradient-outer-product updates, fitted on 3,000, 6,000 and 10,000 labelled rows. Each update uses 12 CG iterations and up to 2,048 gradient samples; 2,000-row kernel panels cap the latter two updates at 2,000 samples.

Each classifier fits kernel ridge on all 10,000 labelled rows with a 256-landmark Nyström preconditioner, followed by a refit that includes confident query rows carrying model-predicted labels. An HKNN affine-hull correction revisits the 512 lowest-margin query rows using 70 neighbours for each of their three candidate classes. The five class-probability vectors are averaged. Query truth is never available to the learner.

The complete fit and prediction sequence runs in a CUDA graph captured during the permitted foreign warm-up. Every replay copies the current inputs, initializes solver and metric state, and refits all five classifiers. Cached objects hold graph execution and scratch storage, not reusable fitted models. The implementation uses plain PyTorch matrix operations and analytic input derivatives, with **no backpropagation**.

## Validation and energy

- **All three initial runs of this exact source passed every canonical gate:** 45 fresh sandboxed timed workers, including 33 MNIST calls and 12 holdout calls. Individual GPU times range from **217.800 to 219.505 ms**. Every timing-dispersion and scorer-clock check passes.
- Source SHA-256 is **`0bb2b7b4852db2bda7726029d7f211d109473d1b7c672a8256d6ce86daa94bef`**, matching every raw record. Source/import validation passes at **11,300 bytes**, with no review flags. Records identify three distinct A100-80GB GPU UUIDs.
- Saved-record verification reconstructs the accuracy, ranked timing and all energy-window verdicts with the unchanged scorer. All three complete unfiltered records are included. No run of the submitted source was dropped or replaced.
- Energy is idle-adjusted **GPU board energy**, less empty-call cost, measured in the canonical 20-second repeated-call window. It excludes host/system energy and warm-up/initial compilation.

These are author-run qualification results submitted for independent maintainer source review and verification. No established-record star is requested before that review.

## Additional held-out validation

The frozen source was also evaluated with the same 10,000-label budget and 9×9 → whitened/rotated 60-dimensional release:

| Test data | Unique query images | Fits | Mean error | Mean GPU ms |
| --- | ---: | ---: | ---: | ---: |
| Standard MNIST official test split | 10,000 | 3 | 2.113333% | 219.220 |
| QMNIST additional test digits | 50,000 | 5 | 2.382000% | 219.255 |

The standard test images are reused across three independently sampled training/release draws. QMNIST uses five disjoint 10,000-query cohorts from indices 10,000–59,999, excluding the reconstruction of the standard test set. All calls use fresh canonical sandbox workers with foreign-data warm-up. No parameters were tuned using these results.

These are external generalization checks, separate from competition qualification. QMNIST's pooled error exceeds 2.30%; this submission does not claim the competition accuracy threshold on that external dataset. Raw results, source hashes and dataset provenance are included.

## Reproduce

From `mnist-a100/`:

```sh
python run_modal.py submissions/rfm-ensemble-d4-20261017/submission.py:kernel_ensemble \
  --difficulty 4 --runs 3 --json /tmp/rfm-ensemble-d4
```

Verify the packaged qualification records:

```sh
python submissions/rfm-ensemble-d4-20261017/verify_records.py \
  submissions/rfm-ensemble-d4-20261017/submission.py \
  submissions/rfm-ensemble-d4-20261017/evidence/official
```

[Source](submission.py), [report](README.md), [raw qualification evidence](evidence/official/), [held-out results](evidence/heldout/).
