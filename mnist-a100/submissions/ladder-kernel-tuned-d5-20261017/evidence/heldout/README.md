# Frozen D5 held-out evaluation

`results.json` contains all eight raw calls, and `summary.json` aggregates them.
`plan.json` fixes the source hash, cohort indices and seeds before results.
`data-manifest.json` pins download URLs and checksums; large dataset files are
not included. `evaluate.py` and `checks.py` are copied unchanged from the tested
pipeline. The qualified learner source is unchanged.

On a root Linux A100 worker with the competition runtime, download the four
files from their pinned URLs into this directory’s `data/` using the manifest
filenames. The evaluation checks every SHA-256 before use. From `mnist-a100/`:

```sh
PYTHONPATH=. python submissions/ladder-kernel-tuned-d5-20261017/evidence/heldout/checks.py
PYTHONPATH=. python submissions/ladder-kernel-tuned-d5-20261017/evidence/heldout/evaluate.py \
  submissions/ladder-kernel-tuned-d5-20261017/submission.py ladder_kernel \
  --output /tmp/d5-heldout-new.json
```

Each call uses a fresh canonical sandbox worker; query truth is retained by the
root scorer. The standard 10k test images are reused across three fresh fits.
The five QMNIST cohorts cover 50k additional query indices without overlap and
exclude the first 10k that reconstruct the standard test set. The test is a
generalization diagnostic, not a competition qualification run or energy test.
