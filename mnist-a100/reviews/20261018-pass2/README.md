# Pull-request review — October 18, 2026, second pass

Merged **#125** (new D3 record, **5.179753 ms / 320.165897 mJ** with the stock canonical launcher) and **#128** (new 16×16 matmul record, **63,026 weighted reads**, 264 fewer than the previous record).

| PR | Independent verification | Decision |
| - | - | - |
| [#125](https://github.com/cybertronai/sutro-problems/pull/125) | 3/3 fresh A100 score and energy passes; margins +213/+307/+271 labels | Merged; [source, report and raw evidence](../../submissions/kernel-d3-20261018/README.md) |
| [#128](https://github.com/cybertronai/sutro-problems/pull/128) | All 256 outputs pass two exact polynomial interpreters; both compute 63,026 reads | Merged; [matmul verification](../../../matmul/submissions/best_63026.md) |
| [#126](https://github.com/cybertronai/sutro-problems/pull/126) | 1/3 fresh A100 passes; complete runs 2/3 miss D4 accuracy by 26/20 labels | Held open; [all outcomes and provenance](pr126/README.md) |
| [#127](https://github.com/cybertronai/sutro-problems/pull/127) | 1/3 fresh A100 passes; complete runs 2/3 miss D5 accuracy by 116/57 labels | Held open; [all outcomes and provenance](pr127/README.md) |

All **nine initial reviewer runs completed all 135 timed calls**. No outcome was replaced or retried. Frozen scorer and launcher files are unchanged, and their hashes are recorded in [canonical integrity](canonical-integrity.json). Learners passed static source review; the two new holds concern actual accuracy reproduction. Each run must independently pass: pooled accuracy across a group is descriptive and cannot repair a failed run. There is no qualifying group time or energy median for #126 or #127; their single passing run is not a group median. Optional energy validity is assessed separately from score qualification.

The #125 author's 3.060774 ms primary-source result uses four CPUs and parent `TMPDIR=/dev`; the accepted leaderboard instead uses the independent stock-launcher result. Historical packaged variants have source-hash mismatches and are retained separately. #126's author records name a different, unsupplied source hash from its submitted learner; their numerical reconstruction does not establish source equivalence. #127's source-matching author 3/3 records reconstruct separately, but the new group supplies contrary accuracy evidence. Author logs and GPU identifiers were removed in all three submissions; the fresh reviewer records remain intact. The accepted D3 report now preserves the identifiable upstream credit chain.

The four older PRs have unchanged heads and no new comments/reviews/evidence resolving their holds: **#114** and **#122** retain accuracy failures (64/4 and 28/74 labels short); **#119** retains two later untimed warmup failures, with 90 seconds implemented versus 180 seconds advertised; **#110** remains explicitly non-record and its requested A100 diagnostic is already complete. No unchanged old source was rerun. [Detailed recheck](older-pr-recheck.json).

Published MNIST A100 instructions specify three official Modal runs and their median. They do not expressly mandate another maintainer batch or permanently ban an unchanged source after a failure; these holds follow contradictory observed reproduction evidence. For #119, the documented and implemented warmup limits remain distinct, and the stalled phase is unresolved.

[Machine-readable results, exact heads and merge commits](summary.json) · [Frozen measurement plan](measurement-plan.json) · [Final GitHub status](final-github-status.json).

[Independent final publication check](independent-final-publication-check.json) replays both held verifiers with expected failure status and checks metrics, hashes, links and winner markers.
