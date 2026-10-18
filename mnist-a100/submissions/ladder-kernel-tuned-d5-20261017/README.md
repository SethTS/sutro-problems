# Tuned Ladder/kernel teacher: difficulty 5

## Independent reviewer reproduction — October 18, 2026

The first frozen reviewer group passed **3/3 fresh canonical sandboxed Modal A100-80GB runs**, with valid above-idle energy measurements on three distinct GPUs. Median ranked time is **897.469072 ms/call** and median energy is **154,288.607829 mJ/call**; pooled MNIST error is **1.813939%** (324,014/330,000 correct).

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 1.759091% | +155 | 902.778503 | 152,205.273845 |
| 2 | 1.817273% | +91 | 896.942848 | 159,265.969906 |
| 3 | 1.865455% | +38 | 897.469072 | 154,288.607829 |

Source is unchanged at SHA-256 `1387f65ce5ff6eb9906f52cca3311cd7be274afa16ceaab1f1d208c68a08b4cb`. The scorer and runner are unchanged. [Verification](evidence/reviewer-20261018/verification.json) reconstructs every complete canonical score and energy result from the [raw records](evidence/reviewer-20261018/); [source review](evidence/reviewer-20261018/source-review.json) covers per-call fitting, label isolation and cached-state reset. No reviewer outcome was replaced or retried.

This is a parameter tuning of @spartanhaden's accepted [Ladder with a kernel teacher](../ladder-kernel-d5-20261012/README.md) (PR #121), submitted by @islamborghini. Independent AST comparison confirms the executable changes are only `BATCH: 1000 → 2000` and `STEPS: 2300 → 800`; architecture, teacher, optimizer, losses and per-call reset remain unchanged, and source credits are preserved. Submitted external-test diagnostics remain separate from competition qualification.

The leaderboard uses the independent reviewer medians above. Author results below are retained separately. The published instructions specify three official Modal runs and their median; this independent reproduction is review evidence, not a newly imposed second-batch submission rule.

Using Python 3.13 as in the canonical runner, recheck saved records from this submission directory with `python evidence/reviewer-20261018/verify_records.py evidence/reviewer-20261018 submission.py --difficulty 5 --function ladder_kernel --output /tmp/pr124-verification.json`. This reconstructs recorded verdicts without executing the learner.

## Original author report (as submitted)

Present-tense record and pending-review claims below describe the submission date; the dated reviewer section above gives the accepted review status.


This tunes **@spartanhaden’s existing [D5 Ladder network with a kernel teacher](https://github.com/cybertronai/sutro-problems/blob/c269e4baa95f2bd96e05dee2107a3330064e8daa/mnist-a100/submissions/ladder-kernel-d5-20261012/README.md)**, introduced in [PR #121](https://github.com/cybertronai/sutro-problems/pull/121). The executable change is limited to **batch size 1,000 → 2,000** and **training steps 2,300 → 800**; the architecture, teacher, optimizer, losses and reset/prediction logic are unchanged. Existing source credits are preserved.

All three initial energy-enabled sandboxed Modal A100-80GB runs of this exact source passed the unchanged scorer (`mnist-a100/1.2.0`): **899.191340 ms/call median**, **155,269.572 mJ/call median above idle**, and **1.790000% pooled MNIST error**. This author-run median takes **50.542% less time (2.022× faster)** than the base entry’s published independent reviewer median of **1,818.087508 ms**; independent maintainer verification of this entry is pending.

## Changes

| Parameter | Base D5 entry | This entry |
| --- | ---: | ---: |
| `BATCH` | 1,000 | 2,000 |
| `STEPS` | 2,300 | 800 |

Step-count documentation is updated as well. The [source diff](parameters.patch) shows the complete change from the base source (SHA-256 `039bde8b8576a67dbfd20017d7b70cbb2fce59f0927d0fc9f989c1bb6db45728`).

## Qualification

Difficulty 5 requires mean MNIST error ≤ **1.90% within each run**, individual-draw error ≤ **3.40%**, and holdout accuracy ≥ **15%**, plus timing and scorer-clock gates.

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Holdout accuracy | Ranked ms | Above-idle mJ |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | A100-SXM4-80GB | 1.750909% | +164 | 1.94% | fashion 88.6075% | 894.564087 | 143,862.207 |
| 2 | A100-SXM4-80GB | 1.825455% | +82 | 2.01% | kmnist 97.1500% | 899.191340 | 155,269.572 |
| 3 | A100-SXM4-80GB | 1.793636% | +117 | 2.23% | fashion 88.5575% | 899.967818 | 161,879.455 |

## Validation

- **3/3 initial runs passed:** all 45 fresh sandboxed timed calls (33 MNIST and 12 holdout calls), with three distinct GPU UUIDs. No run of this submitted source was discarded or replaced.
- The unchanged canonical scorer reconstructs every accuracy, timing-dispersion, scorer-clock and energy-window decision. Ranked times use the slower dataset mean and the parent-clock floor where applicable. Pooled MNIST correctness is **324,093/330,000**; the smallest run margin is **82 labels**.
- Source SHA-256: **`1387f65ce5ff6eb9906f52cca3311cd7be274afa16ceaab1f1d208c68a08b4cb`**, matching every record. The file is **20,411 bytes** and passes the source gate. Its only review flag is proximity to the 20,480-byte size limit.
- Development screened 14 parameter configurations. The selected configuration was frozen before a separate 24-draw confirmation set (seeds 719001–719024), which returned **1.814583% mean error**. Developer screening is separate from official qualification.
- Timed calls include fitting the kernel teacher, initializing and training the student, and prediction. Foreign-data warm-up prepares compilation and CUDA graphs; each scored call refits from scratch. Query features are used transductively without query labels, as in the base entry.
- Energy is idle-adjusted GPU board energy, less empty-call cost; host/system energy and initial compilation are excluded.

These are author-run qualification results submitted for independent maintainer review and verification. No established-record star is requested before that review.

## Additional held-out validation

After selection, the **unchanged frozen source** was evaluated on the official MNIST test split and QMNIST’s additional test digits, using the same 10,000-label training budget and canonical 9×9 → whitened/rotated 60-dimensional release. Each call used a fresh canonical sandbox worker with a foreign Fashion warm-up; full teacher fitting, student initialization/training and prediction were timed.

| Test data | Unique query images | Fits | Mean error | Mean GPU ms |
| --- | ---: | ---: | ---: | ---: |
| Official MNIST test split | 10,000 | 3 | 1.573333% | 898.392 |
| QMNIST additional test digits | 50,000 | 5 | 1.852000% | 898.526 |

The standard test images are reused under three independently sampled 10,000-label training subsets and fresh releases; 30,000 predictions represent 10,000 unique query images. QMNIST uses five disjoint 10,000-query cohorts from indices 10,000–59,999, excluding its reconstruction of the standard test set. No parameters were changed or selected using these results.

Per-fit MNIST errors were **1.60%, 1.49%, 1.63%**. QMNIST cohort errors were **1.74%, 1.64%, 1.94%, 1.92%, 2.02%**: three cohorts individually exceed 1.90%, although the full 50,000-image average is below it. These are external generalization diagnostics, separate from competition qualification; they do not prove absence of overfitting or guarantee the threshold for every future sample. No energy measurement was made for this diagnostic.

[Raw results, cohort plan, dataset provenance and evaluation helper](evidence/heldout/) are included.

## Reproduce

From `mnist-a100/`:

```sh
python run_modal.py submissions/ladder-kernel-tuned-d5-20261017/submission.py:ladder_kernel \
  --difficulty 5 --runs 3 --json /tmp/ladder-tuned-d5
```

Verify the included records:

```sh
python submissions/ladder-kernel-tuned-d5-20261017/verify_records.py \
  submissions/ladder-kernel-tuned-d5-20261017/submission.py \
  submissions/ladder-kernel-tuned-d5-20261017/evidence/official
```

[Source](submission.py), [report](README.md), [raw records and verification](evidence/official/).
