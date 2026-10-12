# Spherical direction/radius kernel — D3

Three original independent energy-enabled canonical sandbox evaluations passed. Median ranked time **51.627899 ms**, median above-idle GPU energy **8,706.872618 mJ**, pooled MNIST error **2.493333%** over 330,000 predictions. Human review remains pending.

| Run | MNIST error | Ranked ms | Above-idle mJ |
|---:|---:|---:|---:|
| run-1.json | 2.440000% | 43.038687 | 8085.754810 |
| run-2.json | 2.471818% | 52.997398 | 8706.872618 |
| run-3.json | 2.568182% | 51.627899 | 8966.888948 |

## Method and attribution

Derived from [PR #116](https://github.com/cybertronai/sutro-problems/pull/116), entrant @islamborghini, whose implementation is credited to DeepSeek V4.1 Flash through OpenCode. Its full RBF solver, 16-step Nyström-preconditioned CG and local class-geometry correction are retained. This submission changes representation/metric learning:

- Normalize the 60-dimensional input direction and retain a separately standardized log-radius channel of weight 0.5.
- Perform two analytic gradient-outer-product metric updates on direction coordinates only. Renormalize the transformed direction after each update, including the normalization Jacobian through tangent projection when computing gradients. Radius remains separate and unlearned.
- Use the spherical chord RBF, not explicit angular coordinates or a complex spectral basis. The radius also contributes to kernel distance. Final local geometry uses the same upstream correction with metric column weights applied only to direction coordinates.

Every call refits from supplied inputs. No external/pretrained data or query labels are used. The local correction uses confident query predictions as pseudo-support for a disjoint uncertain query set, as in the upstream method. Query truth is used only by the scorer. No reconstruction or neural network training is involved.

The measured time is essentially tied with PR #116's reported 51.593931 ms; this is **not a speed-record claim over PR #116**. Its reported error and ours come from different draws, so the canonical pooled errors alone do not establish an accuracy improvement. The paired 11-draw development comparison gave 2.675454% versus 2.701818% for upstream, a small uncertain gain (paired SE 0.029520 percentage points).

## Evidence and provenance

All three original canonical records are retained under `evidence/official`, with a local audit and SHA manifest. Each record contains 11 MNIST calls and four foreign calls, has no disqualification problems, matches the frozen source hash, and passes the energy reconstruction. The energy-enabled records identify three distinct A100 GPU UUIDs. Times include scorer parent-clock floors; energy is idle-adjusted GPU board energy under the canonical repeated-call protocol, excluding host energy and first-call compilation.

Before canonical evaluation, the packaged source matched research predictions exactly on three fresh draws, with repeat/reset and input immutability checks. The source is 16,104 bytes, SHA-256 `43c487a621cd5e563f5636181db4af74816daecfa37c90bf3f850cf699c1df51`.

A separately frozen **one-update Newton–Schulz candidate failed two of three canonical evaluations** (errors 2.720000%, 2.624545%, 2.725455%). Its source and all original records are preserved under `evidence/ns5-failed`; no successful runs from that candidate are pooled into this submission. The spherical two-update candidate is materially different. No failed outcome from its three-run batch was replaced or rerun.

## Reproduce

From `mnist-a100/` (launches paid A100 work):

```sh
python run_modal.py submissions/polar-kernel-d3-20261012/submission.py:classify --difficulty 3 --runs 3 --json /tmp/polar-d3-verification
```

Audit the saved records without GPU work:

```sh
python submissions/polar-kernel-d3-20261012/verify_records.py submissions/polar-kernel-d3-20261012/evidence/official submissions/polar-kernel-d3-20261012/submission.py
```

The audit reconstructs accuracy/timing and energy verdicts and checks source SHA and sandbox status. It is a saved-record check, not an independent GPU reproduction.
