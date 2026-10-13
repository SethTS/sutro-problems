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
