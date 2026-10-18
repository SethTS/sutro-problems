"""Reconstruct official decisions and energy from raw records with the pinned scorer."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import mnist


def verify(source, folder, expected_runs, require_energy, allow_failed=False):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    mnist.check_source(source.read_bytes(), "ladder_kernel", source.name)
    paths = sorted(folder.glob("run-*.json"))
    assert len(paths) == expected_runs, (len(paths), expected_runs)
    summaries, uuids = [], []
    for path in paths:
        raw = json.loads(path.read_text())
        r = raw["record"]
        assert raw["returncode"] in ((0, 1) if allow_failed else (0,)), (path, r)
        assert r["version"] == mnist.VERSION and r["band_bp"] == 190
        assert r["sandboxed"] is True and r["source_sha256"] == digest
        assert "A100" in r["device"] and r["function"] == "ladder_kernel"
        calls = [mnist.Call(**c) for c in r["calls"]]
        assert Counter(c.dataset for c in calls) == Counter({"mnist": 11, r["holdout"]: 4})
        problems, ranked, summary = mnist.judge(calls, r["band_bp"])
        assert problems == r["problems"], (path, problems)
        assert raw["returncode"] == (1 if problems else 0)
        if problems:
            assert allow_failed and r["ranked_ms"] is None, (path, problems)
        else:
            assert math.isclose(ranked, r["ranked_ms"], abs_tol=1e-8)
        digits = [c for c in calls if c.dataset == "mnist"]
        holdout = [c for c in calls if c.dataset != "mnist"]
        energy = r.get("energy")
        if require_energy and not problems:
            assert energy and energy.get("windows"), (path, energy)
        if energy and energy.get("windows"):
            replay = mnist.energy_summary(energy["windows"], energy["device"], 190,
                                          statistics.median(c.ms for c in digits))
            assert replay["problems"] == energy["problems"] == [], (path, replay["problems"])
            assert math.isclose(replay["mj_per_call"], energy["mj_per_call"], abs_tol=1e-6)
            uuids.append(energy["device"]["uuid"])
        summaries.append(dict(run=path.name, passed=not problems, problems=problems,
                              ranked_ms=r["ranked_ms"], unqualified_time_ms=ranked if problems else None,
                              error_pct=100 * (1 - sum(c.correct for c in digits) / sum(c.total for c in digits)),
                              worst_error_pct=max(100 * (1 - c.correct / c.total) for c in digits),
                              holdout=r["holdout"], holdout_accuracy_pct=100 * sum(c.correct for c in holdout) / sum(c.total for c in holdout),
                              energy_mj=energy.get("mj_per_call") if energy else None,
                              gpu_uuid=energy["device"]["uuid"] if energy and energy.get("device") else None,
                              summary=summary))
    passed = sum(r["passed"] for r in summaries)
    qualified = passed == expected_runs
    if require_energy:
        assert len(set(uuids)) == passed, "Passing official runs must use distinct GPUs."
    return dict(source_sha256=digest, runs=summaries, attempted=expected_runs, passed=passed,
                qualifying_three_run_result=qualified and expected_runs == 3,
                median_ranked_ms=statistics.median(r["ranked_ms"] for r in summaries) if qualified else None,
                median_energy_mj=statistics.median(r["energy_mj"] for r in summaries) if qualified and require_energy else None,
                canonical_score_and_energy_reconstructed=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("folder", type=Path)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--no-energy", action="store_true")
    p.add_argument("--allow-failed", action="store_true", help="audit failed attempts without assigning a qualifying median")
    a = p.parse_args()
    result = verify(a.source, a.folder, a.runs, not a.no_energy, a.allow_failed)
    (a.folder / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
