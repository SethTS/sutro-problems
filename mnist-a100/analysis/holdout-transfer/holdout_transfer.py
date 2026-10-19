"""Do the board's winners transfer? Hold-out (KMNIST / Fashion-MNIST) error against MNIST error, by learner family.

Every official run record lists its calls: 11 MNIST and 4 hold-out calls on one hold-out dataset. This script reads
every such record under mnist-a100/ (submissions/ and reviews/, stdlib only), keeps each distinct run once, fits one
line of hold-out error against MNIST error per hold-out dataset, and reports each family's mean offset from that line
within each difficulty band, so that families are compared at matched MNIST accuracy.

usage: python holdout_transfer.py [MNIST_A100_DIR]      (default: the mnist-a100 directory this file sits in)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BANDS = {540: 1, 340: 2, 270: 3, 230: 4, 190: 5}  # band_bp -> difficulty

# Learner family of each submission folder (and, for review folders, of each reviewed PR).
FAMILY = {
    "MLP": ["graph-mlp-20260926", "energy/results"],
    "Ladder": ["ladder-graph-20260928", "ladder-triton-20260928", "ladder-fast-20260929", "ladder-sampled-20260929",
               "ladder-kernel-tuned-20261003"],  # the last is the Ladder with tuned Triton kernels, not a kernel teacher
    "MLP ensemble + kNN readout": ["native-mlp-d1-20261007", "native-mlp-d2-20261007", "native-mlp-d3-20261007"],
    "tangent MLP ensemble": ["tangent-crossteach-d5-20261012", "tangent-relu-d5-20261012", "reviews/20261014/pr122"],
    "kernel, no backprop": ["kernel-pcg-20260929", "kernel-rbf-d3-20261009", "nystrom-fisher-d1-20261012",
                            "rfm-pcg-d2-20261012", "rfm-hknn-d3-20261012", "polar-kernel-d3-20261012",
                            "rfm-ensemble-d4-20261017", "kernel-d3-20261018"],
    "hybrid, backprop + kernel": ["ladder-kernel-d4-20261012", "ladder-kernel-d5-20261012",
                                  "ladder-kernel-tuned-d5-20261017", "reviews/20261018-pass2/pr126",
                                  "reviews/20261018-pass2/pr127"],
}


def family(path, file):
    if file and (file.startswith("ladder") or re.match(r"baseline_d[345]\.py", file)):
        return "Ladder"  # baselines-20260930 and improved-20260930 hold MLPs at d1-d2 and Ladders at d3-d5
    if file and re.match(r"(baseline_d[12]|mlp_d[12]|example|fast_mlp)\.py", file):
        return "MLP"
    for name, folders in FAMILY.items():
        if any(f"/{folder}/" in f"/{path}" for folder in folders):
            return name
    return None


def runs(root):
    seen = {}
    for path in sorted(root.rglob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        record = data.get("record") if isinstance(data, dict) else None
        if not isinstance(record, dict) or not isinstance(record.get("calls"), list):
            continue
        calls = [c for c in record["calls"] if isinstance(c, dict) and "correct" in c]
        mnist = [c for c in calls if c["dataset"] == "mnist"]
        hold = [c for c in calls if c["dataset"] != "mnist"]
        if not mnist or not hold or record.get("band_bp") not in BANDS:
            continue
        err = lambda cs: 100 - 100 * sum(c["correct"] for c in cs) / sum(c["total"] for c in cs)
        rel = path.relative_to(root).as_posix()
        run = {"path": rel, "file": record.get("file"), "band": BANDS[record["band_bp"]], "hold": hold[0]["dataset"],
               "mnist": err(mnist), "holdout": err(hold), "family": family(rel, record.get("file"))}
        key = (record.get("source_sha256"), tuple((c["dataset"], c["correct"]) for c in calls))
        seen.setdefault(key, run)  # the same run is often copied into several evidence folders
    return list(seen.values())


def line(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - slope * mx, slope


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    all_runs = runs(root)
    unknown = sorted({r["path"].rsplit("/", 1)[0] for r in all_runs if r["family"] is None})
    if unknown:
        print("runs with no family (excluded):", *unknown, sep="\n  ")
    rs = [r for r in all_runs if r["family"]]
    families = list(FAMILY)
    print(f"{len(rs)} distinct runs\n")
    for hold in ("kmnist", "fashion"):
        x = [r for r in rs if r["hold"] == hold]
        a, b = line([r["mnist"] for r in x], [r["holdout"] for r in x])
        res = [r["holdout"] - (a + b * r["mnist"]) for r in x]
        sd = (sum(e * e for e in res) / (len(res) - 2)) ** 0.5
        print(f"### {hold}: {len(x)} runs; hold-out error = {a:.2f} + {b:.2f} x MNIST error; residual sd {sd:.2f} pp\n")
        print("| band | " + " | ".join(families) + " |")
        print("|---|" + "---|" * len(families))
        cells = defaultdict(list)
        for r, e in zip(x, res):
            cells[(r["band"], r["family"])].append(e)
        for band in range(1, 6):
            row = []
            for f in families:
                v = cells.get((band, f))
                row.append(f"{sum(v) / len(v):+.2f} ({len(v)})" if v else "")
            print(f"| d{band} | " + " | ".join(row) + " |")
        print()


if __name__ == "__main__":
    main()
