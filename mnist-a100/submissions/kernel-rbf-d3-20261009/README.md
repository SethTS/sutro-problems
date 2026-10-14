# Non-backprop RBF kernel classifier — difficulty 3

RBF kernel ridge with two analytic metric updates, 16-step Nyström-preconditioned CG and local class geometry. CG executes as a CUDA graph with current inputs on every fit. **All six fixed stock runs passed every timing gate.** The predeclared first three runs also passed accuracy and energy; their medians are **49.990213 ms** and **8,417.715 mJ (8.417715 J)**. The complete six-run batch is 5/6 overall: run 6 fails mean accuracy, and remains included below.

Entrant: [@islamborghini](https://github.com/islamborghini). Implementation authored with DeepSeek V4.1 Flash through OpenCode. Entry point: [kernel_d3_cg_graph.py](kernel_d3_cg_graph.py):classify.

## Method

Normalize feature vectors and retain a training-standardized log-radius feature (weight 0.5). Learn two feature metrics from analytic input derivatives on 512 training rows. Fit ten-output RBF kernel ridge on all 10,000 training rows, with gamma 0.02, ridge 0.05, 256 Nyström landmarks and 16 CG steps per solve. Correct the 2,048 lowest-margin queries using 64 neighbors per class, combining labelled training rows and 7,952 disjoint confident queries with predicted labels; add standardized local scores at strength 0.55.

Every call refits its coefficients and metric from the supplied inputs. Query labels are unavailable. The fit uses torch.no_grad and explicit analytic derivatives, with no backward/autograd/optimizer calls or pretrained model. Training Gram storage is FP16; solver/local work is FP32; Triton products use tf32x3.

## Runtime change

The CG operation sequence is captured during the permitted foreign warm-up. Every solve recomputes its Nyström factor, copies the current Gram/targets/factor into static buffers, replays initialization and all 16 iterations, and returns a clone. The graph cache strongly owns its scratch tensors; later replay cannot overwrite weights returned to an earlier caller. Buffers retain their last values between calls but are overwritten before use; no fitted result is reused. Cholesky and eigensolvers remain eager. The successful first fit collects and freezes the warm heap; new-object garbage collection remains enabled.

CUDA graphs reduce timed Python/driver launches. Six passing timing evaluations support this runtime change, but do not establish the cause of an earlier stall or guarantee future cloud timing. No timing padding, omitted computation, altered scorer or modified gate is used.

## Fixed evaluations

Stock scorer 1.2.0, unchanged run_modal.py, CUDA 13.3 / Python 3.13 / Torch 2.12, Modal A100-80GB, sandbox required. Every run has 11 MNIST + 4 foreign calls in fresh workers after foreign warm-up. All 90 timed calls remain in the raw records. The slowest event time is 50.839554 ms. Ranked scores include the scorer's calibrated parent-clock floor.

| Run | Timing | MNIST maximum ms | MNIST ceiling ms | Mean error | Overall | Energy mJ/call |
|---:|---|---:|---:|---:|---|---:|
| 1 | PASS | 49.234943 | 98.096252 | 2.628182% | PASS | 8417.715 |
| 2 | PASS | 49.370113 | 98.643074 | 2.541818% | PASS | 7841.957 |
| 3 | PASS | 50.839554 | 98.303101 | 2.630909% | PASS | 8445.230 |
| 4 | PASS | 47.883263 | 96.709763 | 2.607273% | PASS | 8549.236 |
| 5 | PASS | 41.922562 | 84.219009 | 2.534545% | PASS | 7844.611 |
| 6 | PASS | 49.201153 | 98.528381 | 2.710909% | FAIL (accuracy) | unmeasured |

Runs 1–3 were designated as the qualification group before launch; runs 4–6 were additional checks. Runs 1–3 have pooled MNIST error 2.600303%. Run 6 scored 107,018/110,000 against 107,030 required (12 short); its energy was not attempted after disqualification. The failed run is not replaced or included in a passing-subset qualification claim. All five measured energy results pass the canonical reference/idle/control/accuracy/timing gates. Energy is GPU-board consumption above idle and empty control, excluding setup and host/system energy.

A separate [initial no-energy smoke](evidence/smoke/) also passed (49.526 ms); it is not pooled into the reported medians.

[Complete six-run evidence](evidence/all-six/), [independent arithmetic](evidence/all-six/audit.json), [fixed plan](evidence/official/measurement-plan.json), and [GPU solver comparisons](evidence/numerics.json) bind this exact source. GPU comparisons show zero difference from eager CG in six cases covering changed inputs, zero-target recovery, identical replay, prior-output isolation, odd steps and rank zero; they are developer checks, not qualification runs. All outcomes from the original submitted source remain in [historical evidence](evidence/historical/), with its original source hash and source file.

## Reproduce

From mnist-a100 (launches paid GPU work):

```sh
python run_modal.py submissions/kernel-rbf-d3-20261009/kernel_d3_cg_graph.py:classify --difficulty 3 --runs 6 --json /tmp/kernel-d3-runtime
```

Verify the predeclared saved three-run group locally:

```sh
python submissions/kernel-rbf-d3-20261009/verify_records.py submissions/kernel-rbf-d3-20261009/evidence/official
```

Source SHA-256: 296b69006ad178cf85562387e40d29d8149381a41e1b3c7d7f8bcb1683c565b8; 19,730 bytes (750 below the cap). Scorer SHA-256: 1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064. Runner SHA-256: 1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab. Human acceptance remains pending.
