# Ladder network on learned-metric inputs, with a kernel teacher: difficulty 4

Entrant: [@spartanhaden](https://github.com/spartanhaden).

Three sandboxed Modal A100-80GB runs of the unchanged scorer (`mnist-a100/1.2.0`, `run_modal.py --difficulty 4 --runs 3`) all passed.
- **Median ranked time:** 579.286 ms
- **Median energy above idle:** 67,356 mJ per call

| Run | GPU | MNIST error | Margin (labels) | Worst draw | Hold-out | Ranked ms | mJ above idle |
| ---: | --- | ---: | ---: | ---: | --- | ---: | ---: |
| [1](evidence/official/run-1.json) | A100-SXM4-80GB | 2.108% | +211 | 2.38% | Fashion 88.50% | 575.435 | 69,998 |
| [2](evidence/official/run-2.json) | A100-SXM4-80GB | 2.082% | +240 | 2.26% | KMNIST 96.98% | 579.286 | 67,356 |
| [3](evidence/official/run-3.json) | A100-SXM4-80GB | 2.140% | +176 | 2.39% | Fashion 88.67% | 581.317 | 64,350 |

Compared with the current record (2,882.5 ms, 400,682 mJ; @SethTS, loss-sampled Ladder; open PR #114 reports 985 ms), this entry is **5.0× faster**.

`submission.py` is 20,451 bytes with SHA-256 `f4c69bf486197cd00721320a90dbd644ad734b3371e65ea2fff498d9304ebbdc`, identical to `source_sha256` in all three records. It passes `check_source`. `evidence/official/manifest.json` lists the hash of every file.

## Method

- **Teacher.** The same learned-metric kernel ridge as the d3 entry (two AGOP metric updates and Nyström-preconditioned CG), with a pseudo-label refit whose test-row scores serve as soft labels.
- **Student.** A Ladder network (Rasmus et al. 2015; @SethTS's recipe), 60-1024-512-256-10, trained for 560 Adam steps on inputs mapped by AGOP^0.5. Its loss has three parts: cross-entropy on labelled rows, cross-entropy toward the teacher's soft labels on test rows, and the Ladder's denoising reconstruction cost.
- **Predict.** The clean (noise-free) encoder.
- **Speed.** The training step runs in a CUDA graph with bf16 activations.

## Credits

Ladder network of Rasmus et al. 2015, following @SethTS's [Ladder entries](../ladder-sampled-20260929/README.md) (batch 1,000, bottom denoising cost, loss-proportional labelled sampling), in our own plain-PyTorch implementation. The kernel teacher builds on @islamborghini's [kernel-pcg entry](../kernel-pcg-20260929/README.md).

## Notes

- Transductive: test rows are used without labels, in the unlabelled pool and as distillation targets.
- Training is nondeterministic on the GPU, so reruns on the same draw flip a few dozen labels at about the same error.
- Co-developed with Claude.
