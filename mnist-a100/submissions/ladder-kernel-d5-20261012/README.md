# Ladder network on learned-metric inputs, with a kernel teacher (longer training): difficulty 5

Entrant: [@spartanhaden](https://github.com/spartanhaden).

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, `run_modal.py --difficulty 5 --runs 3`) all passed.
- **Median ranked time:** 1,814.570 ms
- **Median energy above idle:** 233,678 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 1.655% | +270 | 1.82% | Fashion 88.50% | 1,814.570 | 222,845 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 1.668% | +255 | 1.98% | KMNIST 97.52% | 1,814.673 | 233,678 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 1.676% | +246 | 1.91% | Fashion 88.58% | 1,809.458 | 234,856 |

Compared with the current record (3,335.4 ms, 638,234 mJ; @likelytobelaura, cross-taught Muon MLPs; open PR #119 reports 2,581 ms), this entry is **1.8× faster**.

`submission.py` is 20,414 bytes with SHA-256 `039bde8b8576a67dbfd20017d7b70cbb2fce59f0927d0fc9f989c1bb6db45728`, identical to `source_sha256` in all three records. It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

- **Shared design.** Same design as the d4 entry, trained for 2,300 Adam steps.
- **Inputs.** The student uses the teacher's AGOP^0.75 metric coordinates.
- **Compile.** Cheap-compile inductor settings keep the cold compile inside the scorer's warm-up limit.

## Credits

Ladder network of Rasmus et al. 2015, following @SethTS's [Ladder entries](../ladder-sampled-20260929/README.md), in our own plain-PyTorch implementation. The kernel teacher builds on @islamborghini's [kernel-pcg entry](../kernel-pcg-20260929/README.md).

## Notes

- Transductive: test rows are used without labels, in the unlabelled pool and as distillation targets.
- Training is nondeterministic on the GPU, so reruns on the same draw flip a few dozen labels at about the same error.
- Co-developed with Claude.

## Independent maintainer verification — 2026-10-12

All three initial fresh energy-enabled sandboxed Modal A100-80GB runs passed individually. Source review found no qualification blocker. All 45 timed calls and complete energy windows were reconstructed with the unchanged canonical scorer; three distinct GPU UUIDs were recorded. No failed run was replaced or learner repair made.

The reviewer medians are **1818.087508 ms** and **226351.850269 mJ above idle per call**. The leaderboard uses this independent set; the author source and original evidence above are retained.

| Reviewer run | Correct | MNIST error | Ranked ms | Above-idle mJ |
|---:|---:|---:|---:|---:|
| 1 | 108,014/110,000 | 1.805455% | 1818.087508 | 226351.850269 |
| 2 | 108,148/110,000 | 1.683636% | 1811.331920 | 231912.536637 |
| 3 | 108,215/110,000 | 1.622727% | 1818.706696 | 221130.907800 |

[Raw records and manifest](evidence/reviewer-20261012/) and [canonical reconstruction](evidence/reviewer-20261012/verification.json) are retained. Learner SHA-256: `039bde8b8576a67dbfd20017d7b70cbb2fce59f0927d0fc9f989c1bb6db45728`. Energy excludes host energy and first-call compilation.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/ladder-kernel-d5-20261012/submission.py:ladder_kernel --difficulty 5 --runs 3 --json /tmp/pr121-d5-verification
```
