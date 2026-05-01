# Findings: Scratchpad Layout Optimization for Tiled 16×16 Matmul

**Date:** 2026-04-30  
**Author:** Seth Stafford  
**Score:** 110,743 (-17.2% vs tiled_16x16 baseline of 133,783)  
**IR:** [`ir/tiled_16x16_opt1.ir`](ir/tiled_16x16_opt1.ir)

---

## Hypothesis

In `generate_tiled_16x16`, the `tmp` register sits at address 49 (cost 7) but is read on every multiply-accumulate step. Moving it to address 1 (cost 1) should save substantially.

## Method

Profiled read counts for every scratchpad slot in the tiled 16×16 algorithm:

| Slot | Reads | Reason |
|------|------:|-------|
| `tmp` | 4,096 | read after every `mul` (once per inner loop iteration: nb³ × T³ = 4,096) |
| each `sA(ii,kk)` cell | 256 | read T times per (bi,bj,bk) block × nb³ blocks |
| each `sB(kk,jj)` cell | 256 | same |
| each `sC(ii,jj)` cell | 256 | 15 accumulation reads + 1 writeback read per (bi,bj) × nb² |

`tmp` is 16× hotter than any other slot. Since `_cost(addr) = ⌈√addr⌉` is monotone, the optimal assignment is greedy: place the hottest slot at the lowest address.

The 48 `sA`/`sB`/`sC` cells have identical read counts, so their ordering among addresses 2–49 does not affect total cost.

## New Layout (opt1)

| Region | Addresses | Change |
|--------|-----------|--------|
| `tmp`  | 1         | was 49 (cost 7 → 1) |
| `sA`   | 2–17      | was 1–16 |
| `sB`   | 18–33     | was 17–32 |
| `sC`   | 34–49     | was 33–48 |
| A/B/C bulk | 50–817 | unchanged |

## Results

| Method | Cost | vs baseline |
|--------|-----:|------------|
| `generate_tiled_16x16` (baseline) | 133,783 | — |
| `generate_tiled_16x16_opt1` (tmp@1) | 110,743 | **-17.2%** |

## Why This Is Optimal for This Algorithm

The total scratchpad read cost is:

```
cost = Σ_slot  reads(slot) × _cost(addr(slot))
```

With uniform reads across the 48 non-tmp slots, minimizing this sum requires only that tmp lands at address 1 and the remaining 48 slots occupy addresses 2–49 in any order. No further permutation can reduce cost.

## Remaining Cost Structure

After opt1, the dominant costs are:

1. **sC accumulation reads** — `add sC(ii,jj), tmp` reads `sC` on 3,840 of 4,096 inner iterations. Eliminating this would require a fused multiply-add instruction not present in v0.
2. **Bulk A/B copy-load reads** — 512 elements each read 4 times at addresses 50+. Cannot be reduced without changing the input/output interface.
3. **sA/sB/sC inner-loop reads** — 48 cells × 256 reads at addresses 2–49. These are already at the lowest available addresses.

Further improvement requires algorithmic changes: different tiling, loop reordering, or instruction set extensions.
