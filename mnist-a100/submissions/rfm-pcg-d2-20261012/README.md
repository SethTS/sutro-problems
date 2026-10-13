# RBF kernel ridge with a learned metric, preconditioned CG: difficulty 2

Entrant: [@spartanhaden](https://github.com/spartanhaden).

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, `run_modal.py --difficulty 2 --runs 3`) all passed.
- **Median ranked time:** 4.736 ms
- **Median energy above idle:** 570 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 3.069% | +364 | 3.23% | Fashion 87.50% | 4.533 | 570 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 2.997% | +443 | 3.31% | Fashion 87.53% | 5.092 | 601 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 3.133% | +294 | 3.32% | Fashion 87.11% | 4.736 | 370 |

Compared with the current record (14.6 ms, 1,333 mJ; @sjbaebae, native MLPs with three-neighbor voting), this entry is **3.1× faster**.

`submission.py` is 12,480 bytes with SHA-256 `d2e38e1072b2f7ca6e0e3ade3746d2bddb32604aca8d1ddcc6127233135f2776`, identical to `source_sha256` in all three records. It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

- **Metric learning (one RFM step).** Fit a small kernel ridge model on 2,000 train rows. The average outer product of its class-score gradients with respect to the inputs (the AGOP) gives a 60×60 metric, and every row is mapped by its Cholesky factor.
- **Fit.** Kernel ridge on all 10,000 rows in the learned metric, with 5 conjugate-gradient steps preconditioned by a rank-512 Nyström approximation. Stopping early regularises.
- **Predict.** The argmax of the test rows' kernel scores.
- **Speed.** The 10,000 × 10,000 kernel is never formed: kernel–vector products run through PyTorch's fp16 flash-attention kernel (an exact identity for the RBF kernel). The call is one CUDA graph, with the preconditioner built on a side stream.

## Credits

Builds directly on @islamborghini's [kernel-pcg entry](../kernel-pcg-20260929/README.md) (RBF kernel ridge, one AGOP metric update, Nyström-preconditioned CG). This file re-tunes it and rebuilds it in plain PyTorch. The AGOP metric step is from Recursive Feature Machines (Radhakrishnan et al. 2022, arXiv 2212.13881).

## Notes

- No backprop: the metric uses analytic input gradients of a closed-form model, and the fits are a few CG steps.
- Inductive: test rows are only scored.
- Plain PyTorch with no `torch.compile`, so the warm-up is short and the scorer's post-warm-up staging cost stays small.
- Co-developed with Claude.

## Independent maintainer verification — 2026-10-12

All three initial fresh energy-enabled sandboxed Modal A100-80GB runs passed individually. Source review found no qualification blocker. All 45 timed calls and complete energy windows were reconstructed with the unchanged canonical scorer; three distinct GPU UUIDs were recorded. No failed run was replaced or learner repair made.

The reviewer medians are **4.578529 ms** and **602.662112 mJ above idle per call**. The leaderboard uses this independent set; the author source and original evidence above are retained.

| Reviewer run | Correct | MNIST error | Ranked ms | Above-idle mJ |
|---:|---:|---:|---:|---:|
| 1 | 106,444/110,000 | 3.232727% | 4.382246 | 602.662112 |
| 2 | 106,529/110,000 | 3.155455% | 4.578529 | 628.332434 |
| 3 | 106,558/110,000 | 3.129091% | 5.892114 | 425.005391 |

[Raw records and manifest](evidence/reviewer-20261012/) and [canonical reconstruction](evidence/reviewer-20261012/verification.json) are retained. Learner SHA-256: `d2e38e1072b2f7ca6e0e3ade3746d2bddb32604aca8d1ddcc6127233135f2776`. Energy excludes host energy and first-call compilation.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/rfm-pcg-d2-20261012/submission.py:classify --difficulty 2 --runs 3 --json /tmp/pr121-d2-verification
```
