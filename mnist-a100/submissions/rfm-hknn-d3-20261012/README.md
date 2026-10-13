# Learned-metric kernel ridge with HKNN on low-margin rows: difficulty 3

Entrant: [@spartanhaden](https://github.com/spartanhaden).

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, `run_modal.py --difficulty 3 --runs 3`) all passed.
- **Median ranked time:** 18.391 ms
- **Median energy above idle:** 2,218 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 2.429% | +298 | 2.64% | KMNIST 96.54% | 18.193 | 2,301 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 2.399% | +331 | 2.65% | Fashion 88.16% | 19.228 | 2,218 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 2.485% | +236 | 2.63% | KMNIST 96.62% | 18.391 | 2,181 |

Compared with the current record (51.8 ms, 8,839 mJ; @sjbaebae, spherical metric kernel (#117); #116 reports 51.6 ms), this entry is **2.8× faster**.

`submission.py` is 14,707 bytes with SHA-256 `da456d0ed99d8fb59b26e76b1658d6cd8d69b31710122936f2ca5f81f0ced966`, identical to `source_sha256` in all three records. It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

- **Metric learning, twice.** Fit RBF kernel ridge to the labels (on 3,000 and then 6,000 rows). Take the AGOP of the fitted scores, and warp the inputs by about AGOP^0.75.
- **Fit.** Kernel ridge on all 10,000 rows in the final metric, using 13 Nyström-preconditioned CG steps.
- **HKNN correction.** For the 512 lowest-margin test rows only, measure the distance to each of the top-3 classes' local affine hull: the ridge-regularised hull of the 70 nearest same-class vertices. Vertices are train rows plus confident test rows carrying their predicted label (HKNN: Vincent & Bengio 2002). These distances add a small probability term to the kernel scores.
- **Speed.** fp16 kernel panels with a hi/lo operand split, test scores built in L2-sized chunks, and the preconditioner on a side stream. It all runs as one CUDA graph, in eager mode with no `torch.compile`.

## Credits

Builds on @islamborghini's [kernel-pcg entry](../kernel-pcg-20260929/README.md) and Recursive Feature Machines (Radhakrishnan et al. 2022). The local-hull step is HKNN (Vincent & Bengio, NeurIPS 2002).

## Notes

- No backprop: closed-form linear algebra, a few CG steps and analytic input gradients.
- Transductive: test rows are used without labels, as HKNN vertices carrying model-predicted labels.
- Our earlier, slower version of this pipeline (with a pseudo-label refit, 52.1 ms on Modal on 2026-10-05) showed that the refit added little once HKNN is present. This version drops it.
- Co-developed with Claude.

## Independent maintainer verification — 2026-10-12

All three initial fresh energy-enabled sandboxed Modal A100-80GB runs passed individually. Source review found no qualification blocker. All 45 timed calls and complete energy windows were reconstructed with the unchanged canonical scorer; three distinct GPU UUIDs were recorded. No failed run was replaced or learner repair made.

The reviewer medians are **18.143665 ms** and **2361.815112 mJ above idle per call**. The leaderboard uses this independent set; the author source and original evidence above are retained.

| Reviewer run | Correct | MNIST error | Ranked ms | Above-idle mJ |
|---:|---:|---:|---:|---:|
| 1 | 107,400/110,000 | 2.363636% | 18.411804 | 2100.016333 |
| 2 | 107,256/110,000 | 2.494545% | 18.112284 | 2361.815112 |
| 3 | 107,212/110,000 | 2.534545% | 18.143665 | 2370.329065 |

[Raw records and manifest](evidence/reviewer-20261012/) and [canonical reconstruction](evidence/reviewer-20261012/verification.json) are retained. Learner SHA-256: `da456d0ed99d8fb59b26e76b1658d6cd8d69b31710122936f2ca5f81f0ced966`. Energy excludes host energy and first-call compilation.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/rfm-hknn-d3-20261012/submission.py:rfm_d3 --difficulty 3 --runs 3 --json /tmp/pr121-d3-verification
```
