"""Block C/D labeler: turn SWE-bench-Live evaluation reports into real failure
labels on the captured trajectories, replacing the exit-code proxy.

Pipeline:
  1. Capture trajectories + predictions with scripts/06_run_batch.py
     (-> data/features/<id>.parquet, data/predictions/<split>.json).
  2. Run SWE-bench-Live's evaluator on the predictions (its own repo + Docker):
       python -m evaluation.evaluation \
         --dataset SWE-bench-Live/SWE-bench-Live --split <split> --platform linux \
         --patch_dir <abs path>/data/predictions/<split>.json \
         --output_dir <abs path>/logs/eval --workers 4
     Each resolved instance gets logs/eval/<instance_id>/report.json with
     {"resolved": bool, "FAIL_TO_PASS": {...}, "PASS_TO_PASS": {...}}.
  3. This script reads those reports and rewrites each features parquet:
       will_fail        = not resolved
       steps_to_failure = [n-1, ..., 0] if will_fail else <null>
     Any captured trajectory without a resolved=True report (empty patch,
     overflow, limits, or a wrong submitted patch) is labeled will_fail=True.

Safe to run before the evaluator exists: with no reports it labels everything
will_fail=True (a correct all-failure pass), and re-running after eval flips the
resolved ones. Idempotent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def resolved_instances(eval_dir: Path) -> set[str]:
    """Instance ids with resolved=True in logs/eval/<id>/report.json."""
    resolved: set[str] = set()
    for report in eval_dir.glob("*/report.json"):
        try:
            data = json.loads(report.read_text())
        except Exception:
            continue
        if data.get("resolved") is True:
            resolved.add(str(data.get("instance_id", report.parent.name)))
    return resolved


def relabel_parquet(path: Path, resolved: bool) -> tuple[int, bool]:
    """Rewrite will_fail / steps_to_failure on one features parquet in place."""
    df = pd.read_parquet(path)
    n = len(df)
    will_fail = not resolved
    df["will_fail"] = will_fail
    df["steps_to_failure"] = list(range(n - 1, -1, -1)) if will_fail else pd.NA
    df.to_parquet(path, index=False)
    return n, will_fail


def main() -> None:
    ap = argparse.ArgumentParser(description="Relabel Plumb features from SWE-bench-Live eval reports")
    ap.add_argument("--eval-dir", default="logs/eval", help="output_dir passed to the SWE-bench-Live evaluator")
    ap.add_argument("--features-dir", default="data/features")
    ap.add_argument("--exclude", default="toy-,mini-", help="comma-separated instance-id prefixes to skip")
    args = ap.parse_args()

    eval_dir = Path(args.eval_dir)
    resolved = resolved_instances(eval_dir)
    if not eval_dir.exists():
        print(f"[label] note: {eval_dir} not found -> labeling all captured trajectories as failures")
    print(f"[label] resolved={len(resolved)} {sorted(resolved)}")

    skip = tuple(p for p in args.exclude.split(",") if p)
    rows: list[tuple[str, int, bool]] = []
    for fp in sorted(Path(args.features_dir).glob("*.parquet")):
        iid = fp.stem
        if iid.startswith(skip):
            continue
        n, wf = relabel_parquet(fp, iid in resolved)
        rows.append((iid, n, wf))

    print(f"\n{'instance':44}{'steps':>6}{'will_fail':>10}")
    for iid, n, wf in rows:
        print(f"{iid:44}{n:>6}{str(wf):>10}")
    n_fail = sum(1 for _, _, wf in rows if wf)
    print(f"\n[label] {len(rows)} trajectories | will_fail={n_fail} | resolved(success)={len(rows) - n_fail}")


if __name__ == "__main__":
    main()
