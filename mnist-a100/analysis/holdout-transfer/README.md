# Do the board's winners transfer?

A first pass at mining the mnist-a100 board: is "kernels beat backprop" an MNIST artefact? Each official run
record also scores the learner on four hold-out calls (KMNIST or Fashion-MNIST, same release pipeline), so the
board already contains the evidence. Nothing here was run on a GPU.

**Short answer: no MNIST artefact shows up.** Hold-out error tracks MNIST error closely (residual sd ≈ 0.45 pp),
and at matched MNIST error the kernel entries transfer as well as or better than the backprop entries. The
plain Ladder transfers worst to Fashion-MNIST.

## Method

[`holdout_transfer.py`](holdout_transfer.py) (stdlib only) reads every run record under `mnist-a100/`
(`submissions/` and `reviews/`), keeps each distinct run once (records are often copied into several evidence
folders), and labels it with a learner family by submission folder. Per hold-out dataset it fits one line of
hold-out error against MNIST error over all runs, then reports each family's mean offset from that line within
each band. Negative means the family does better on the hold-out than its MNIST error predicts.

Families: **MLP** (single or ensemble, softmax readout); **Ladder**; **MLP ensemble + kNN readout** (@sjbaebae's
native MLPs); **tangent MLP ensemble** (@likelytobelaura's cross-taught Muon MLPs); **kernel, no backprop**
(learned-metric RBF kernel ridge and its descendants); **hybrid, backprop + kernel** (Ladder taught by a kernel,
and the two held reviewer runs of #126/#127, whose sources use both autograd and Cholesky solves).

## Results (main at `f66769e`, 193 distinct runs)

Mean offset in percentage points, runs in brackets.

**KMNIST** (104 runs; hold-out error = 0.12 + 1.41 × MNIST error)

| band | MLP | Ladder | MLP ens. + kNN | tangent MLP ens. | kernel, no backprop | hybrid |
|---|---|---|---|---|---|---|
| d1 | +0.85 (6) |  | −1.26 (3) |  | −0.06 (5) |  |
| d2 | +0.51 (5) |  | −0.15 (2) |  | +0.04 (6) |  |
| d3 |  | +0.23 (13) | +0.36 (9) |  | −0.40 (26) |  |
| d4 |  | +0.10 (5) |  |  | −0.24 (3) | +0.09 (5) |
| d5 |  | +0.05 (5) |  | +0.00 (9) |  | +0.05 (2) |

**Fashion-MNIST** (89 runs; hold-out error = 10.31 + 0.75 × MNIST error)

| band | MLP | Ladder | MLP ens. + kNN | tangent MLP ens. | kernel, no backprop | hybrid |
|---|---|---|---|---|---|---|
| d1 | −0.05 (9) |  |  |  | −0.56 (1) |  |
| d2 | −0.39 (3) |  | −0.32 (1) |  | −0.24 (9) |  |
| d3 |  | +0.59 (7) | −0.16 (1) |  | +0.32 (17) |  |
| d4 |  | +0.38 (6) |  |  | −0.51 (3) | −0.10 (7) |
| d5 |  | +0.12 (6) |  | −0.41 (6) |  | −0.24 (13) |

Readings:

- **No family's lead on MNIST is bought at the cost of transfer.** Every offset is within about 1.3 pp, most
  within 0.5 pp, so the ranking would likely hold on these two sister datasets.
- **Kernels transfer well.** Better than the line on KMNIST at d3 and d4; on Fashion-MNIST worse at d3, better at d4.
- **The plain Ladder is the weakest on Fashion-MNIST** at d3–d4. Taught by a kernel, the same architecture moves
  to the better side there (neutral on KMNIST), so the teacher seems to add something that generalises, not only
  speed.
- **The MLP + kNN readout looks unusually good on KMNIST at d1**, but on 3 runs.

## Caveats

- Each run has 40,000 hold-out predictions, and many cells have 1–6 runs: read the pattern across bands, not any
  single cell.
- Hold-out accuracies are not calibrated against the bands (as noted when the hold-outs were introduced), which is
  why families are compared at matched MNIST error and raw hold-out scores are never compared.
- Family offsets are confounded with band (no family appears in every band), so compare within a row.
- Both hold-outs are close cousins of MNIST (28×28 greyscale, same release). This says nothing about transfer to
  genuinely different data.
- Family labels are by folder; see `FAMILY` in the script. PR branches that never reached main are not included.

## Reproduce

```bash
python mnist-a100/analysis/holdout-transfer/holdout_transfer.py
```

[`results.md`](results.md) is the script's output at `f66769e`.
