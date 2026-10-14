# D5: cross-taught Muon ReLU-MLP ensemble with local tangent jitter (fast)

## Independent reviewer reproduction — October 14, 2026

The first frozen reviewer group passed **3/3 fresh canonical sandboxed Modal A100-80GB runs**, with valid above-idle energy measurements on three distinct GPUs. Median ranked time is **1,915.402496 ms/call** and median energy is **343,151.622557 mJ/call**; pooled MNIST error is **1.789091%** (324,096/330,000 correct).

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 1.810000% | +99 | 1,922.228505 | 331,163.170874 |
| 2 | 1.812727% | +96 | 1,913.758972 | 343,151.622557 |
| 3 | 1.744545% | +171 | 1,915.402496 | 346,421.252166 |

Source is unchanged at SHA-256 `9adc570a02279b34d0766344ec72f5589ff10fae06e1b6b341b58dc351a6445a`. The scorer and runner are unchanged. [Verification](evidence/reviewer-20261014/verification.json) reconstructs every complete canonical score and energy result from the [raw records](evidence/reviewer-20261014/); [source review](evidence/reviewer-20261014/source-review.json) covers per-call fitting, label isolation and cached-state reset. No reviewer outcome was replaced or retried.

The leaderboard uses the independent reviewer medians above. Author results below are retained separately. The published instructions specify three official Modal runs and their median; this independent reproduction is review evidence, not a newly imposed second-batch submission rule.

Using Python 3.13 as in the canonical runner, recheck saved records from this submission directory with `python evidence/reviewer-20261014/verify_records.py evidence/reviewer-20261014 submission.py --output /tmp/pr120-verification.json`. This reconstructs recorded verdicts without executing the learner.

## Original author report (as submitted)

Present-tense record and pending-review claims below describe the submission date; the dated reviewer section above gives the accepted review status.


Entrant: [@likelytobelaura](https://github.com/likelytobelaura). The joint train+query graph-propagation readout is
[@sjbaebae](https://github.com/sjbaebae)'s, from the D3 record (see [Credits](#credits)).

Three canonical Modal A100-80GB sandbox evaluations, the first and only set run for this source, **all passed
difficulty 5** (mean MNIST error at most 1.90%). Median ranked time is **1,898.373 ms per call**; median energy above
idle is **333,939.8 mJ per call**. The 33 MNIST calls labelled **324,016 / 330,000** rows correctly
(**1.813% aggregate error**). This is the same method as the current D5 record
([tangent-crossteach-d5-20261012](../tangent-crossteach-d5-20261012/README.md), 3,335.4 ms) with faster members and
a faster readout: 43% less time per call at about half the energy.

| Run | GPU | MNIST error | Margin, labels | Hold-out | Ranked ms | mJ above idle | Record |
|---|---|---:|---:|---|---:|---:|---|
| 1 | A100 80GB PCIe | 1.809% | +100 | Fashion 88.64% | 1,916.868 | 333,940 | [json](official/run-1.json) |
| 2 | A100-SXM4-80GB | 1.810% | +99 | Fashion 88.78% | 1,894.813 | 339,773 | [json](official/run-2.json) |
| 3 | A100-SXM4-80GB | 1.821% | +87 | KMNIST 97.22% | **1,898.373** | 321,951 | [json](official/run-3.json) |

The three runs used three distinct GPU UUIDs. The worst single MNIST draw was 2.15% (the cap is band + 1.5 = 3.40%),
and every call was within 1.5% of its dataset's median time. [verification.json](official/verification.json)
recomputes every verdict, the ranked times and the per-call gates from the raw per-call data in the records.

Source: `submission.py`, 13,363 bytes, SHA-256 `9adc570a02279b34d0766344ec72f5589ff10fae06e1b6b341b58dc351a6445a`
(identical to the `source_sha256` in all three records). `mnist.check_source` passes and `mnist.review_flags`
reports nothing. The file is source only: scalar hyperparameters, no arrays, no fitted data.

## Method

- **Members.** Eight MLPs 60-256-256-256-10, batched over members. Each hidden layer is `x W^T`, then RMSNorm with no
  learnable gain, then ReLU, then dropout 0.1; hidden layers have no bias. First layer initialised from normalised
  training rows. 2,200 updates of 2,048 labelled rows (+1,024 pseudo-labelled rows once cross-teaching starts),
  cosine schedule, EMA of the weights.
- **Optimizer.** Muon (momentum 0.95, 3-step Newton-Schulz in BF16, all square matrices in one batched call) on every
  hidden matrix; AdamW on the head. Weight decay 0.1.
- **Local tangent jitter.** Once per call, from the draw's own labelled rows: for each row, the offsets to its 10
  nearest same-class rows give a 10x10 Gram matrix whose top 4 eigenvectors yield the row's 4 local tangent
  directions, each scaled to the rms offset along it. Every update moves each sampled labelled row by an N(0, 1)
  step along each of its tangents, then applies mixup (Beta(0.5, 0.5), partner = next row of the batch).
- **In-graph dropout.** Dropout masks come from a 32-bit counter hash of (element index, update number, layer),
  computed inside the compiled step, so no random tensors are materialised; checked to give independent
  Bernoulli(0.9) masks across elements, layers and updates.
- **Cross-teaching on disjoint query pools.** The members form two halves of four; the query rows are split once
  into two pools. From 20% of training, every 200 updates, each half reads out all query rows, and the 90% it is
  surest of (members' mean probability of the label) become hard pseudo-labels for the other half, which trains on
  them only within its own pool.
- **Readout.** Centred last-hidden EMA embeddings, signed power 0.25 per coordinate, L2-normalised, concatenated
  over members. A 5-nearest-labelled-neighbour prior (temperature 20) feeds three propagation steps over the kNN
  graph of labelled rows (clamped to their labels) plus query rows (k 5, alpha 0.9). This propagation readout,
  algorithm and constants, is @sjbaebae's from the D3 record. New here: the signed-power embedding, the hub
  correction in the final readout (each row's mean similarity to its 10 nearest query rows, x0.3, subtracted), and
  computing neighbours as two 10,000 x 10,000 blocks (query x labelled, query x query) whose top-k lists are merged,
  which gives the same neighbours as one 10,000 x 20,000 matrix at lower cost.
- **Engineering.** Training steps are CUDA graphs (loss/backward and Muon+EMA compiled by Inductor), captured during
  the warm-up call. Every call recomputes the tangents and re-initialises all weights, optimizer state, EMA,
  pseudo-labels and pools.

## Rules review

Checked against the rules in `mnist.py`'s docstring:

- **Trains from scratch on every call.** `classify` runs `_load` (copies the new draw and recomputes its tangents)
  and `_reset` (weights, EMA, Muon momentum, AdamW state, update counter, pseudo-labels, pools, init generator seed)
  before any update. The warm-up call only compiles and captures graphs; nothing learned on it reaches the timed
  call, and every call runs in a fresh process anyway.
- **Uses only `train_x`, `train_y`, `test_x`.** Query rows are used without labels (pseudo-labels, kNN graph), as
  in the D3 record. Query labels never enter the method.
- **Invariant to the release.** Classes are handled symmetrically, so the secret label permutation is irrelevant;
  tangents, neighbours and the data-sampled init are rotation-invariant functions of the draw.
- **No fitted data or offline prior.** Only scalar hyperparameters, chosen on MNIST development draws (seeds
  401-416 of `mnist.draw`), scored outside the method; nothing derived from those draws is in the file. The
  per-layer bias and gain tensors are allocated but unused (they stay at their reset values 0 and 1).
- **Sandbox and files.** All three runs were sandboxed. The only side effect is pointing the Triton and Inductor
  caches at `/tmp` (set at call time), which the scorer clears between calls.
- **Same work per call.** Fixed update count and cross-teaching schedule; the slowest call per dataset was at most
  1.015x its median.

## Development (A100, `dev/quick.py`, seeds 401-416, not official scores)

| Step | Mean error | ms per call |
|---|---:|---:|
| GELU MLPs with gain and bias, 2,500 updates ([current D5 record](../tangent-crossteach-d5-20261012/README.md), PR #118) | 1.803% | 3,349 |
| + in-graph hash dropout, fused tangent jitter, block readout | 1.787-1.804% | 2,950-2,977 |
| + ReLU, no hidden gain or bias | 1.842% | 2,187 |
| + 2,200 updates (this file) | 1.838-1.849% | 1,893-1,949 |

Removing the gain and bias takes the norm block from 143 to 47 us per layer (their gradients are column sums over
all rows, which Inductor cannot fuse with the row-wise norm backward). With these members, a re-tune of tangent
strength, Muon learning rate, weight decay, cross-teaching coverage and cadence, dropout and member count found
nothing beyond noise.

## Credits

- Joint train+query kNN graph propagation readout (5-nearest-labelled-neighbour prior, then three propagation
  steps over the kNN graph of labelled and query rows, k 5, alpha 0.9): by [@sjbaebae](https://github.com/sjbaebae),
  from the D3 record ([graph1250-repair](../native-mlp-d3-20261007/graph1250-repair/README.md)). This submission
  reuses it unchanged except for the embedding it runs on, the block-wise neighbour computation and the
  final-readout hub correction.
- Local tangent jitter, cross-teaching on disjoint query pools, Muon training, ReLU members without hidden gain or
  bias, in-graph hash dropout, signed-power embeddings and hub correction:
  [@likelytobelaura](https://github.com/likelytobelaura). The tangents follow the tangent-distance / tangent-prop
  line of work by Simard et al.
- Muon and its Newton-Schulz coefficients: Keller Jordan et al.

## Reproduce

From `mnist-a100`:

```bash
python run_modal.py submissions/tangent-relu-d5-20261012/submission.py:classify --difficulty 5 --runs 3 --json out/d5-tangent-relu
```
