# PR #126 — held after independent review

The initial fixed group for the submitted source passed **1/3** canonical sandboxed Modal A100-80GB score runs. All **45 timed calls** completed. Runs 2 and 3 failed mean MNIST accuracy; no outcome was retried or replaced. The learner remains **open and unmerged**, with **no qualifying time or energy median**.

| Run | MNIST correct / 110,000 | Error | Margin against 107,470 required | Ranked ms | Above-idle mJ | Score |
| - | -: | -: | -: | -: | -: | - |
| [1](run-1.json) | 107,523 | 2.251818% | +53 | 34.923388 | 5,759.762546 | pass |
| [2](run-2.json) | 107,444 | 2.323636% | -26 | — | — | fail: accuracy |
| [3](run-3.json) | 107,450 | 2.318182% | -20 | — | — | fail: accuracy |

The descriptive pooled error is **2.297879%** (322,417/330,000 correct). Although below 2.30%, it cannot combine separate failed runs into a qualifying group: each run must pass D4. Run 1 has valid energy; runs 2 and 3 did not attempt energy after score disqualification. Energy is an optional column, separate from score eligibility. Every timing and holdout gate passed.

The submitted source is SHA-256 `1966a800debdb3681b1c897527c28771d958a02604b0281b9dbe8508f183aa77` (18,558 bytes), at head `4199965260948ad7837fadbaccc7639b02cb80d6`. Scorer and runner are unchanged from main `14a92c25a899f8ae4509c0068a6ce577b508fc1b`. The [raw outcomes](.), [verification](verification.json), [decision](review-decision.json), [source review](source-review.json), and independent [parent verification](parent-verification.json) retain the complete result.

Author records are preserved separately in [author-records](author-records/). They identify a different source, SHA-256 `c7b78944bfbc45859ec53ec68b94d0808a93decec0c4e2115ae6b6e3c52d78b1` (18,611 bytes), which is not supplied. Their numerical and energy results reconstruct, but equivalence to the submitted source cannot be established. Console logs and GPU UUIDs were stripped from those author records. They do not replace any reviewer failure.

The kernel helpers match @spartanhaden's [Ladder/kernel teacher](../../../submissions/ladder-kernel-d4-20261012/README.md) after alias expansion and local-variable renaming. The neural design follows @likelytobelaura's [tangent ReLU/Muon](../../../submissions/tangent-relu-d5-20261012/README.md) and [tangent cross-teaching](../../../submissions/tangent-crossteach-d5-20261012/README.md) entries, which credit @sjbaebae for the joint train+query graph-propagation readout. This identifies code/design lineage, not whole-source equivalence.

With Python 3.13, from `mnist-a100/`, supply the exact pinned source as `/path/to/submission.py` and run:

```sh
python reviews/20261018-pass2/pr126/verify_records.py \
  reviews/20261018-pass2/pr126 /path/to/submission.py \
  --difficulty 4 --function classify --record-filename submission.py \
  --output /tmp/pr126-recheck.json
```

Expected exit status: **1**, with 1/3 score passes and null qualifying medians. This reads saved records and never executes the learner. The SHA-256 inventory excludes its own manifest.

[Exact unmerged learner source](https://github.com/cybertronai/sutro-problems/blob/4199965260948ad7837fadbaccc7639b02cb80d6/mnist-a100/submissions/neural-d4-20261018/submission.py)
