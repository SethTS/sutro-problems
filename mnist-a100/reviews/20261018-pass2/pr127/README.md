# PR #127: held after accuracy reproduction

Reviewed head: `19d2a3faba642814aee9c12ed1a118a927b796d2`. The [exact submitted source](https://github.com/cybertronai/sutro-problems/blob/19d2a3faba642814aee9c12ed1a118a927b796d2/mnist-a100/submissions/ladder-d5-20261018/submission.py) has SHA-256 `5754027efaadd7b3b2b26ea5e1f7bd1c953c6e70b1692797887565e6e23ba005`; the learner is not copied into this review directory.

No static qualification blocker was found. The PR remains held because the new independent reviewer group reproduced only **1/3 passes**. **All 45 timed calls completed**: 33 MNIST and 12 foreign holdout calls. D5 requires at least 107,910 correct predictions in each run's 110,000 MNIST predictions.

| Run | MNIST correct | Error | Margin | Result |
| --- | ---: | ---: | ---: | --- |
| 1 | 107,969 | 1.846364% | +59 | pass |
| 2 | 107,794 | 2.005455% | −116 | accuracy failure |
| 3 | 107,853 | 1.951818% | −57 | accuracy failure |

Pooled error is 1.934545%; pooling cannot override either failed run. **No qualifying time or energy median exists.** Run 1 alone measured 308.575192 ms and 36,438.337155 mJ above idle. Energy is an optional separate column; these failures are accuracy failures. Runs 2 and 3 therefore have no energy measurement.

The separate [author evidence](https://github.com/cybertronai/sutro-problems/tree/19d2a3faba642814aee9c12ed1a118a927b796d2/mnist-a100/submissions/ladder-d5-20261018) reconstructs as 3/3 source-matching passes; its sanitized records are preserved under `author/`. The author omitted console logs and GPU identifiers, so those records cannot establish distinct boards or original attempt history. They do not replace the contradictory reviewer outcomes. No reviewer run was retried, discarded or replaced. This hold is not a claim of permanent source ineligibility or an unambiguous source-rule violation.

The source audit identifies substantial optimizer, kernel, noise and readout changes by @sigkillme0, with 11 teacher helpers matching @spartanhaden's [PR #121 upstream](https://github.com/cybertronai/sutro-problems/blob/14a92c25a899f8ae4509c0068a6ce577b508fc1b/mnist-a100/submissions/ladder-kernel-d5-20261012/submission.py) after documented alias/local-name normalization. Existing upstream credits include @islamborghini's kernel-PCG and @SethTS/Rasmus et al.'s Ladder work. Submitted source/README omit that chain; documentation would need repair before promotion. See `source-review.json` and `upstream-attribution-audit.json`.

## Reproduce the saved-record check

From the repository root, fetch the pinned source and replay the records with the trusted repository scorer. This runs no GPU evaluations:

```sh
curl -fsSL https://raw.githubusercontent.com/cybertronai/sutro-problems/19d2a3faba642814aee9c12ed1a118a927b796d2/mnist-a100/submissions/ladder-d5-20261018/submission.py -o /tmp/pr127-submission.py
python mnist-a100/reviews/20261018-pass2/pr127/verify_records.py \
  mnist-a100/reviews/20261018-pass2/pr127 /tmp/pr127-submission.py \
  --function ladder_kernel --difficulty 5 --record-filename submission.py \
  --output /tmp/pr127-record-verification.json
```

Expected exit status: **1**, with `passed_runs: 1` and both medians `null`. The explicit record filename binds normalized record metadata while the source SHA-256 binds content. `verify_records.py` loads `mnist-a100/mnist.py` and `run_modal.py`; `packaged-verifier-check.json` records the successful reproduction of this failed group. `evidence-sha256.json` hashes every packaged file except itself.
