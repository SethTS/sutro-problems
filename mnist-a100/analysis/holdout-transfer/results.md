193 distinct runs

### kmnist: 104 runs; hold-out error = 0.12 + 1.41 x MNIST error; residual sd 0.46 pp

| band | MLP | Ladder | MLP ensemble + kNN readout | tangent MLP ensemble | kernel, no backprop | hybrid, backprop + kernel |
|---|---|---|---|---|---|---|
| d1 | +0.85 (6) |  | -1.26 (3) |  | -0.06 (5) |  |
| d2 | +0.51 (5) |  | -0.15 (2) |  | +0.04 (6) |  |
| d3 |  | +0.23 (13) | +0.36 (9) |  | -0.40 (26) |  |
| d4 |  | +0.10 (5) |  |  | -0.24 (3) | +0.09 (5) |
| d5 |  | +0.05 (5) |  | +0.00 (9) |  | +0.05 (2) |

### fashion: 89 runs; hold-out error = 10.31 + 0.75 x MNIST error; residual sd 0.42 pp

| band | MLP | Ladder | MLP ensemble + kNN readout | tangent MLP ensemble | kernel, no backprop | hybrid, backprop + kernel |
|---|---|---|---|---|---|---|
| d1 | -0.05 (9) |  |  |  | -0.56 (1) |  |
| d2 | -0.39 (3) |  | -0.32 (1) |  | -0.24 (9) |  |
| d3 |  | +0.59 (7) | -0.16 (1) |  | +0.32 (17) |  |
| d4 |  | +0.38 (6) |  |  | -0.51 (3) | -0.10 (7) |
| d5 |  | +0.12 (6) |  | -0.41 (6) |  | -0.24 (13) |

