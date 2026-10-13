# D5: cross-taught Muon MLP ensemble with local tangent jitter

Entrant: [@likelytobelaura](https://github.com/likelytobelaura). The joint train+query graph-propagation readout is
[@sjbaebae](https://github.com/sjbaebae)'s, from the D3 record (see [Credits](#credits)).

Three canonical Modal A100-80GB sandbox evaluations, the first and only set run for this source, **all passed
difficulty 5** (mean MNIST error at most 1.90%). Median ranked time is **3,359.445 ms per call**; median energy
above idle is **677,766.5 mJ per call**. The 33 MNIST calls labelled **324,184 / 330,000** rows correctly
(**1.762% aggregate error**).

| Run | GPU | MNIST error | Margin, labels | Hold-out | Ranked ms | mJ above idle | Record |
|---|---|---:|---:|---|---:|---:|---|
| 1 | A100 80GB PCIe | 1.790% | +121 | KMNIST 97.63% | 3,369.836 | 633,927 | [json](official/run-1.json) |
| 2 | A100-SXM4-80GB | 1.748% | +167 | Fashion 88.93% | **3,359.445** | **677,766** | [json](official/run-2.json) |
| 3 | A100-SXM4-80GB | 1.749% | +166 | KMNIST 97.52% | 3,352.078 | 699,197 | [json](official/run-3.json) |

The three runs used three distinct GPU UUIDs. The worst single MNIST draw was 2.06% (the cap is band + 1.5 =
3.40%), and every call was within 0.7% of its dataset's median time. [verification.json](official/verification.json)
recomputes every verdict, the ranked times and the per-call gates from the raw per-call data in the records.

Source: `submission.py`, 12,135 bytes, SHA-256 `a3b18bac64dddc3f7aa8c614e2279932aa74040720cb861e8dfa40eaa67923c3`
(identical to the `source_sha256` in all three records). `mnist.check_source` passes and `mnist.review_flags`
reports nothing. The file is source only: about 35 scalar hyperparameters, no arrays, no fitted data.

## Method

- **Members.** Eight MLPs 60-256-256-256-10 (RMSNorm, then GELU, dropout 0.1), batched over members, first layer
  initialised from normalised training rows. 2,500 updates of 2,048 labelled rows (+1,024 pseudo-labelled rows
  once cross-teaching starts), cosine schedule, EMA of the weights.
- **Optimizer.** Muon (momentum 0.95, 3-step Newton-Schulz in BF16, all square matrices in one batched call) on
  every hidden matrix; AdamW on gains, biases and head. Weight decay 0.1.
- **Local tangent jitter (new).** Once per call, from the draw's own labelled rows: for each row, the offsets to its
  10 nearest same-class rows give a 10x10 Gram matrix whose top 4 eigenvectors yield the row's 4 local tangent
  directions, each scaled to the rms offset along it. Every update moves each sampled labelled row by an N(0, 1)
  step along each of its tangents, then applies mixup (Beta(0.5, 0.5), partner = next row of the batch). This
  replaces the D4 entry's Gaussian input noise. Query rows are not jittered.
- **Cross-teaching on disjoint query pools.** The members form two halves of four; the query rows are split once
  into two pools. From 20% of training, every 200 updates, each half reads out all query rows, and the 90% it is
  surest of (members' mean probability of the label) become hard pseudo-labels for the other half, which trains on
  them only within its own pool.
- **Readout.** Centred last-hidden EMA embeddings, signed power 0.25 per coordinate, L2-normalised, concatenated
  over members. A 5-nearest-labelled-neighbour prior (temperature 20) feeds three propagation steps over the kNN
  graph of labelled rows (clamped to their labels) plus query rows (k 5, alpha 0.9). This propagation readout,
  algorithm and constants, is @sjbaebae's from the D3 record; the signed-power embedding and the hub correction
  below are new here. In the final readout only,
  similarities are hub-corrected: each row's mean similarity to its 10 nearest query rows, x0.3, is subtracted.
- **Engineering.** Training steps are CUDA graphs (loss/backward and Muon+EMA compiled by Inductor), captured during
  the warm-up call. Every call recomputes the tangents and re-initialises all weights, optimizer state, EMA,
  pseudo-labels and pools.

## Rules review

Checked against the rules in `mnist.py`'s docstring:

- **Trains from scratch on every call.** `classify` runs `_load` (copies the new draw and recomputes its tangents)
  and `_reset` (weights, EMA, Muon momentum, AdamW state, step counter, pseudo-labels, pools, init generator seed)
  before any update. The warm-up call only compiles and captures graphs; nothing learned on it reaches the timed
  call, and every call runs in a fresh process anyway.
- **Uses only `train_x`, `train_y`, `test_x`.** Query rows are used without labels (pseudo-labels, kNN graph),
  as in the D3 record and the D4/D5 entries before it. Query labels never enter the method.
- **Invariant to the release.** Classes are handled symmetrically, so the secret label permutation is irrelevant;
  tangents, neighbours and the data-sampled init are rotation-invariant functions of the draw.
- **No fitted data or offline prior.** Only scalar hyperparameters. They were chosen on MNIST development draws
  (seeds 401-416 of `mnist.draw`), scored outside the method; nothing derived from those draws is in the file.
- **Sandbox and files.** All three runs were sandboxed. The only side effect is pointing the Triton and Inductor
  caches at `/tmp` (set at call time, as the D4 entry does), which the scorer clears between calls.
- **Same work per call.** Fixed update count, fixed cross-teaching schedule; the slowest call per dataset was at
  most 1.007x its median.

## Development (A100, `dev/quick.py`, not official scores)

| Step | Mean error, seeds 401-408 | ms per call |
|---|---:|---:|
| D4 cross-taught ensemble (1,700 updates) | 2.149% | 2,170 |
| + signed-power embeddings, temperature 20, final-readout hub correction | 2.070% | 2,163 |
| + local tangent jitter, strength 1.0 | 1.847% | 2,343 |
| + 2,500 updates | 1.801% | 3,383 |
| + no input noise | 1.785% | 3,383 |
| This file, seeds 401-416 | 1.793% | 3,497 |

With tangent jitter on, these were neutral or worse: middle-layer embeddings, 16 members, width 384, more than
2,500 updates, 5 or 20 neighbours, 2 or 8 directions, no mixup, and tangent jitter of query rows (label-free or
along pseudo-class neighbours).

## Credits

- Joint train+query kNN graph propagation readout (5-nearest-labelled-neighbour prior, then three propagation
  steps over the kNN graph of labelled and query rows, k 5, alpha 0.9): by [@sjbaebae](https://github.com/sjbaebae),
  from the D3 record ([graph1250-repair](../native-mlp-d3-20261007/graph1250-repair/README.md)). This submission
  reuses it unchanged except for the embedding it runs on and the final-readout hub correction.
- Local tangent jitter, cross-teaching on disjoint query pools, Muon training, signed-power embeddings and hub
  correction: [@likelytobelaura](https://github.com/likelytobelaura). The tangents follow the tangent-distance /
  tangent-prop line of work by Simard et al.
- Muon and its Newton-Schulz coefficients: Keller Jordan et al.

## Reproduce

From `mnist-a100`:

```bash
python run_modal.py submissions/tangent-crossteach-d5-20261012/submission.py:classify --difficulty 5 --runs 3 --json out/d5-tangent
```

## Independent maintainer verification — 2026-10-12

All three initial fresh energy-enabled canonical Modal A100-80GB runs passed. Source review found no compliance blocker: caller inputs are copied into owned buffers, tangents are fitted anew, and weights, EMA, optimizer state, pseudo-labels and query pools are reset on every call. All 45 timed calls completed and three distinct GPU UUIDs were recorded. No failed outcome was replaced, and no learner retry or repair was made.

The reviewer set has median ranked time **3335.394886 ms**, median above-idle GPU board energy **638234.154568 mJ**, and descriptive pooled MNIST error **1.824848%**. Each run passes the 1.90% gate individually. The top-level leaderboard uses this reviewer set; author and research records remain separate.

| Reviewer run | Correct | MNIST error | Ranked ms | Above-idle mJ |
|---:|---:|---:|---:|---:|
| 1 | 107,940/110,000 | 1.872727% | 3335.394886 | 662154.992877 |
| 2 | 108,063/110,000 | 1.760909% | 3358.272278 | 629924.834404 |
| 3 | 107,975/110,000 | 1.840909% | 3322.150457 | 638234.154568 |

[Raw records and frozen manifest](evidence/reviewer-20261012/), [canonical reconstruction](evidence/reviewer-20261012/verification.json), and [source review](evidence/reviewer-20261012/source-review.json) are retained. The learner SHA remains `a3b18bac64dddc3f7aa8c614e2279932aa74040720cb861e8dfa40eaa67923c3`.

Verify the reviewer records locally from `mnist-a100/`:

```sh
python submissions/tangent-crossteach-d5-20261012/evidence/reviewer-20261012/verify_records.py \
  submissions/tangent-crossteach-d5-20261012/evidence/reviewer-20261012 \
  submissions/tangent-crossteach-d5-20261012/submission.py \
  --output /tmp/d5-reviewer-verification.json
```

Energy is above-idle GPU board energy under the canonical repeated-call protocol, excluding host/system energy and first-call compilation. Comparisons with other submissions use separate draws and devices, not a paired accuracy experiment.
