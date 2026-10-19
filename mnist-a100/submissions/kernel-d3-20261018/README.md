# a100 d3

## Independent reviewer reproduction — October 18, 2026, second pass

The first frozen reviewer group passed **3/3 fresh canonical sandboxed Modal A100-80GB runs**, with valid above-idle energy measurements (3 recorded GPU UUIDs). Median ranked time is **5.179753 ms/call** and median energy is **320.165897 mJ/call**; pooled MNIST error is **2.460303%** (321,881/330,000 correct).

| Reviewer run | MNIST error | Margin, labels | Ranked ms | Above-idle mJ |
| - | -: | -: | -: | -: |
| 1 | 2.506364% | +213 | 4.782926 | 237.266407 |
| 2 | 2.420909% | +307 | 5.179753 | 320.165897 |
| 3 | 2.453636% | +271 | 5.224783 | 322.379107 |

Source is unchanged at SHA-256 `df1158db88162d674dd869998f0795aedee96bb088f506803c5370bc872b3e39`. The scorer and runner are unchanged. [Verification](evidence/reviewer-20261018-pass2/verification.json) reconstructs every complete canonical score and energy result from the [raw records](evidence/reviewer-20261018-pass2/); [source review](evidence/reviewer-20261018-pass2/source-review.json) covers per-call fitting, label isolation and cached-state reset. No reviewer outcome was replaced or retried.

This submission builds on @spartanhaden's accepted [RFM/HKNN D3 learner](../rfm-hknn-d3-20261012/README.md), including its partial normalization, learned-metric RBF/PCG and margin-based local-hull readout. This also inherits @islamborghini's [kernel-PCG implementation](../kernel-pcg-20260929/README.md) and the HKNN local-hull method (Vincent & Bengio, 2002), as credited by Haden. New work includes fused Triton PCG, kernel compression, Fisher/QDA metric preprocessing and revised readout. The independent group above uses the unchanged canonical launcher and its default CPU/temp-directory settings. The author's 3.060774 ms primary-source group instead reserved four CPUs and set parent `TMPDIR=/dev`; that group is separate evidence. Historical `default.py` and `cpu4.py` records name different source hashes from their packaged files and do not establish qualification of those exact packaged variants. All author records omit console logs and GPU identifiers, so their original attempt history and board identities cannot be independently recovered.

The leaderboard uses the independent reviewer medians above. Author results below are retained separately. The published instructions specify three official Modal runs and their median; this independent reproduction is review evidence, not a newly imposed second-batch submission rule.

Using Python 3.13 as in the canonical runner, recheck saved records from this submission directory with `python evidence/reviewer-20261018-pass2/verify_records.py evidence/reviewer-20261018-pass2 submission.py --difficulty 3 --function rfm_d3 --output /tmp/pr125-verification.json`. This reconstructs recorded verdicts without executing the learner.

## Original author report (as submitted)

Present-tense record and pending-review claims below describe the submission date; the dated reviewer section above gives the accepted review status.


author measurements: three passing sandboxed a100-80gb runs per source. [results](results.json).

| source | median ms | pooled error |
| --- | ---: | ---: |
| [default](default.py) | 5.495054 | 2.574242% |
| [cpu4](cpu4.py) | 5.437366 | 2.541515% |
| [cpu4-dev](submission.py) | 3.060774 | 2.510303% |

`default` uses stock launcher settings; `cpu4` reserves four cpu cores; `cpu4-dev` also sets `TMPDIR=/dev`.

gpu reproduction from `mnist-a100/`, with modal configured:

```sh
python submissions/kernel-d3-20261018/run_cpu4_dev.py submissions/kernel-d3-20261018/submission.py:rfm_d3 \
  --difficulty 3 --runs 3 --json /tmp/d3-cpu4-dev

python run_modal.py submissions/kernel-d3-20261018/default.py:rfm_d3 \
  --difficulty 3 --runs 3 --json /tmp/d3-default

python submissions/kernel-d3-20261018/run_cpu4.py submissions/kernel-d3-20261018/cpu4.py:rfm_d3 \
  --difficulty 3 --runs 3 --json /tmp/d3-cpu4
```

offline check from this directory: `python3 -B verify_records.py`. [records](evidence/) omit console logs and gpu identifiers; filenames normalized.
