# Non-backprop RBF kernel classifier — difficulty 3, 2026-10-12

RBF kernel ridge classification with two analytic feature-metric updates and a local class-geometry correction. **All three fixed energy-enabled sandboxed Modal A100-80GB runs of unchanged scorer 1.2.0 passed, including every energy check; time and energy are their respective medians.**

**Submission for maintainer review.**

| Difficulty | Band | Metric updates | CG steps per solve | ms/call | mJ/call above idle | MNIST error, 33 calls | File |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | --- |
| 3 | 2.70% | 2 | 16 | **51.593931** | **8,499.142** | **2.592121%** | [kernel_d3_two_metric_low_ridge.py](kernel_d3_two_metric_low_ridge.py) |

The 33 MNIST calls predicted **321,446 / 330,000** labels correctly. Energy is GPU board energy per call above idle, less the calibrated empty-call cost. The median is **8.499142 J per call**. It excludes host/system energy and first-call compilation.

## Method and attribution

Entrant: [@islamborghini](https://github.com/islamborghini). Implementation authored with **DeepSeek V4.1 Flash through OpenCode**. The entry point is `classify(train_x, train_y, test_x)`.

1. Normalize each supplied feature vector to a common radius, retaining a training-standardized log-radius feature with weight 0.5.
2. Fit an RBF kernel predictor and evaluate its input derivatives analytically on 512 training rows. Form their average gradient outer-product metric, normalize its trace, regularize it toward identity and take its matrix square root. Perform two such metric updates.
3. Fit the final ten-output kernel ridge system. RBF gamma is 0.02 and ridge is 0.05. Each of the two metric fits and the final fit uses 16 conjugate-gradient steps with a 256-landmark Nyström preconditioner. The model uses the full training Gram; the landmarks accelerate the solve.
4. Rank query confidence by the margin between the top two class scores. Correct the 2,048 least-confident queries using 64 nearest supports per class. Each class support combines labelled training rows with model-generated pseudo-labels from the 7,952 more-confident queries. The two query sets are disjoint. A regularized local covariance score is standardized across classes and added to the kernel score with strength 0.55.

The local correction uses supplied query features transductively. Query labels are unavailable to the learner; pseudo-labels come only from its current kernel predictions. Every call constructs all fitted coefficients, metrics, supports and workspace anew, including after the permitted foreign warm-up. There are no pretrained coefficients, external examples or retained fitted state.

The complete fit runs under `torch.no_grad()`. Kernel input derivatives are explicit formulas and matrix operations; there are no backward/autograd/optimizer calls. The method is backprop-free, rather than derivative-free.

## GPU implementation

Triton kernels implement RBF construction, split-K ten-output matrix products, conjugate-gradient reductions/vector updates, and fused local-neighbourhood moments. Local covariance solves use batched Cholesky operations. Training Gram storage is FP16; solver state and local covariance work use FP32. Triton dot products explicitly use `tf32x3`. These fixed precision choices apply to every fit; the PyTorch matmul TF32 flag is False.

## Runs

Difficulty 3 requires mean MNIST error at most **2.70% across the 11 MNIST calls within each run**. No individual MNIST draw may exceed **4.20%**, the band plus 1.50 percentage points. Every recorded run satisfies both limits.

| Run | GPU / power limit | Ranked ms/call | Above-idle mJ/call | MNIST correct | Mean error | Worst error | Holdout accuracy | Raw evidence |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | A100-SXM4-80GB, 400 W | 50.127887 | 8,947.584 | 107,140/110,000 | 2.600000% | 3.03% | kmnist: 97.0525% | [JSON](evidence/official/run-1.json) |
| 2 | A100-SXM4-80GB, 400 W | 51.593931 | 8,499.142 | 107,044/110,000 | 2.687273% | 2.96% | kmnist: 96.7625% | [JSON](evidence/official/run-2.json) |
| 3 | A100-SXM4-80GB, 400 W | 53.217094 | 7,971.171 | 107,262/110,000 | 2.489091% | 2.64% | kmnist: 96.9525% | [JSON](evidence/official/run-3.json) |

Each run contains 11 MNIST and four foreign calls, each in a fresh sandboxed worker after foreign warm-up. The unchanged stock Modal runner used three single-use containers. Energy records contain three distinct GPU UUIDs. All three fixed energy-enabled outcomes are included; no failed outcome was replaced or rerun. Ranked time includes the scorer's parent-clock floor and the slower dataset mean.

Run 2 has the narrowest accuracy margin: 14 correct labels above the required aggregate. These observed passes do not establish future qualification probability. Original raw prediction arrays and remote execution-time scorer hashes are not retained by the stock runner.

## Energy measurement

The canonical protocol reads NVML board-energy counters around a 20-second method window, subtracts adjacent frozen-worker idle power, and subtracts the empty round-trip energy per call. A separate FP32 matrix-multiplication reference checks telemetry before importing the learner. Every reference, idle, context, energy-window accuracy and timing-consistency check passed.

| Run | Reference J/TFLOP | Reference TFLOP/s | Mean idle W | Mean method-window W | Method-window calls | Method-window seconds |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8.3941 | 18.0900 | 87.785 | 285.341 | 442 | 20.047865 |
| 2 | 8.2281 | 18.1854 | 70.058 | 228.156 | 372 | 20.020592 |
| 3 | 7.9960 | 18.0758 | 63.756 | 228.426 | 413 | 20.030671 |

[Energy reconstruction](evidence/official/energy-audit.json) recalculates the results directly from the raw windows, then compares them with unchanged `energy_summary`; all three values match. The fixed [measurement plan](evidence/official/measurement-plan.json) was recorded before the energy batch. This is a board-energy measurement under the canonical repeated-call protocol, not whole-system energy or per-kernel energy.

Each energy window starts with 32 prebuilt image draws and then reuses them under fresh rotations and label permutations. The 1,227 energy-window calls are repeated measurements, not 1,227 independent image draws or additional qualification runs.

## Settings and validation

The configuration was fixed before the three stock runs. An additional development-only pooled error target of 2.60% was missed at 2.637222%; that target remains unmet. Before observing official outcomes, advancement to the fixed three-run stock check used the published 2.70% gate. This submission claims the published requirements, not the additional development buffer.

- Exact learner bytes pass the stock source/import checker at 20,388 bytes, 92 bytes below the limit. All records identify the packaged source hash and `classify` entry point.
- All mean/worst MNIST accuracy, foreign accuracy, output, sandbox, timing-dispersion and independent parent-clock gates pass. Saved canonical `judge` reconstructions reproduce every verdict and ranked time.
- Separate source/evidence reviews read the complete learner, traced query-truth isolation and per-call state, and independently reconstructed all 45 numerical call records. No concrete source/protocol violation was found. These are saved-record reviews, not new GPU reproductions or maintainer acceptance.
- The evaluation scorer and runner remain unchanged. All three canonical energy/reference/control/idle checks pass, and numerical reconstructions match the raw records.

## Reproduce

From `mnist-a100/`, with Modal configured; this launches paid A100 work:

```sh
python run_modal.py submissions/kernel-rbf-d3-20261009/kernel_d3_two_metric_low_ridge.py:classify \
  --difficulty 3 --runs 3 --json /tmp/kernel-rbf-d3-energy
```

Verify the packaged saved records locally, without GPU work:

```sh
python submissions/kernel-rbf-d3-20261009/verify_records.py \
  submissions/kernel-rbf-d3-20261009/evidence/official \
  --source submissions/kernel-rbf-d3-20261009/kernel_d3_two_metric_low_ridge.py
```

[Verification output](evidence/official/verification.log) reports every saved-run check and the three-run median. [Summary](evidence/official/summary.json) contains reconstructed aggregates. [Manifest](evidence/official/manifest.json) binds the exact source and unfiltered records.

## Initial no-energy verification

The first fixed three-run stock verification remains preserved separately in [raw no-energy evidence](evidence/no-energy/), with its [manifest](evidence/no-energy/manifest.json), [summary](evidence/no-energy/summary.json) and [verification output](evidence/no-energy/verification.log). All three passed: ranked times 57.024201, 52.809442 and 49.038230 ms; mean errors 2.666364%, 2.610909% and 2.582727%. That set's median was 52.809442 ms and pooled error 2.6200%. It contains no energy measurements. It is not pooled into the energy-enabled time or energy medians above. The new batch was requested to measure the missing energy column, with unchanged learner bytes.

## SHA-256 and provenance

- `kernel_d3_two_metric_low_ridge.py`: `13dfbe8b83a2cd92c422e576ff4b0b8b756c440f8132b61beb4ffa821b144dc4`.
- Scorer 1.2.0: `1bc6d8d96a776565bc70f3ccdb91be7de57187d4495b2b0437fac98f5e120064`.
- Stock Modal runner: `1180b843cf6c32414774da90c835a8b7298d2db8719989237a793ca1c62a10ab`.
- All three raw-record hashes are preserved in the manifest. These local scorer/runner identities matched current upstream during review; the original stock record format does not attest their hashes inside the remote execution.

No scorer, runner, dataset release or evaluation gate is changed by this submission. Human source review and acceptance remain pending.
