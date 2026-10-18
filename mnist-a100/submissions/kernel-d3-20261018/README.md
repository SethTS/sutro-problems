# a100 d3

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
