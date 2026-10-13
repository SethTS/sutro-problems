# Nyström kernel ridge with a Fisher metric: difficulty 1

Entrant: [@spartanhaden](https://github.com/spartanhaden).

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, `run_modal.py --difficulty 1 --runs 3`) all passed.
- **Median ranked time:** 2.154 ms
- **Median energy above idle:** 63 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 4.949% | +496 | 5.35% | KMNIST 93.11% | 2.245 | 63 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 5.005% | +434 | 5.37% | KMNIST 92.66% | 2.154 | 63 |
| [3](evidence/official/run-3.json) | A100 80GB PCIe | 4.910% | +539 | 5.22% | Fashion 86.55% | 1.603 | 99 |

Compared with the current record (6.7 ms, 530 mJ; @sjbaebae, native MLPs with three-neighbor voting), this entry is **3.1× faster**.

`submission.py` is 18,323 bytes with SHA-256 `34ac5441dec258a0f9fc49f42c58d865e96020146df0a42b062305f8bf25346b`, identical to `source_sha256` in all three records. It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

- **Fisher metric (closed form).** The inputs are whitened, so the within-class scatter is I minus the between-class part. Its inverse comes from the class means by the Woodbury identity, and maps rows into a space where Euclidean distance is the LDA (Mahalanobis) distance.
- **Landmarks.** 48 centres per class (480 in all), refined by two class-restricted k-means steps.
- **Features.** RBF similarities to the 480 landmarks, plus the linear coordinates and 45 quadratic (LDA-like) terms.
- **Readout.** Nyström kernel ridge regression, solved with one Cholesky factorisation. Prediction is the argmax of the class scores.
- **Speed.** The whole call is one CUDA graph, with fp16 tensor-core GEMMs. Because the A100's fp16 Gram matrix can lose positive definiteness on some draws, a graph-safe fallback Cholesky with a larger ridge is selected by `torch.where`.

## Credits

Nyström kernel approximation; Fisher/LDA metric; class-wise k-means (competitive-learning) landmarks.

## Notes

- No backprop and no iterative training: everything is closed-form linear algebra plus two k-means steps.
- Inductive: test rows are only featurised and scored.
- An earlier attempt at this run set failed while Modal built the scorer image (a dataset download was reset), before the method ran. The three runs here are the first complete set for this source.
- Co-developed with Claude.

## Independent maintainer verification — 2026-10-12

All three initial fresh energy-enabled sandboxed Modal A100-80GB runs passed individually. Source review found no qualification blocker. All 45 timed calls and complete energy windows were reconstructed with the unchanged canonical scorer; three distinct GPU UUIDs were recorded. No failed run was replaced or learner repair made.

The reviewer medians are **1.627842 ms** and **60.207195 mJ above idle per call**. The leaderboard uses this independent set; the author source and original evidence above are retained.

| Reviewer run | Correct | MNIST error | Ranked ms | Above-idle mJ |
|---:|---:|---:|---:|---:|
| 1 | 104,573/110,000 | 4.933636% | 2.149163 | 60.207195 |
| 2 | 104,600/110,000 | 4.909091% | 1.627842 | 60.501830 |
| 3 | 104,474/110,000 | 5.023636% | 1.497903 | 58.883797 |

[Raw records and manifest](evidence/reviewer-20261012/) and [canonical reconstruction](evidence/reviewer-20261012/verification.json) are retained. Learner SHA-256: `34ac5441dec258a0f9fc49f42c58d865e96020146df0a42b062305f8bf25346b`. Energy excludes host energy and first-call compilation.

Reproduce from `mnist-a100/`:

```sh
python run_modal.py submissions/nystrom-fisher-d1-20261012/submission.py:nys_d1 --difficulty 1 --runs 3 --json /tmp/pr121-d1-verification
```
