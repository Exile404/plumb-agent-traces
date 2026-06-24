"""Assemble all per-step feature parquets into one modeling dataset with
within-trajectory targets (free, single-agent; needs no successes).

Per-step targets:
  next_step_error   : will step t+1's observation contain an error?   (binary)
  loop_next         : will step t+1 be a repeated_state? (loop onset)  (binary)
  steps_to_collapse : steps remaining until the run ends (== steps_to_failure)
Trajectory-level:
  exit_status / failure_mode  (from data/runs.json; falls back to preds)

Writes data/dataset.parquet and prints class balances.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FEATURES_DIR = Path("data/features")
RUNS = Path("data/runs.json")
PREDS = Path("data/predictions/lite.json")
OUT = Path("data/dataset.parquet")
SKIP = ("toy-", "mini-")


def failure_mode(exit_status: str, patch_chars: int) -> str:
    s = (exit_status or "").lower()
    if "contextwindow" in s:
        return "overflow"
    if "limitsexceeded" in s or "timeexceeded" in s:
        return "stall"
    if "submitted" in s:
        return "wrong_fix" if patch_chars > 0 else "wrong_target"
    if s in ("", "none"):
        return "wrong_fix" if patch_chars > 0 else "unknown"
    return s  # RepeatedFormatError or other exception names


def main() -> None:
    runs = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    preds = json.loads(PREDS.read_text()) if PREDS.exists() else {}
    frames = []
    for fp in sorted(FEATURES_DIR.glob("*.parquet")):
        iid = fp.stem
        if iid.startswith(SKIP):
            continue
        df = pd.read_parquet(fp).sort_values("step_index").reset_index(drop=True)
        df["next_step_error"] = df["error_in_observation"].shift(-1)
        df["loop_next"] = df["repeated_state"].shift(-1)
        fallback = pd.Series(range(len(df) - 1, -1, -1), index=df.index)
        stc = df["steps_to_failure"] if "steps_to_failure" in df.columns else fallback
        df["steps_to_collapse"] = pd.to_numeric(stc, errors="coerce").fillna(fallback)
        # --- trend features: within-run step-over-step change (causal) ---
        for col in ("mean_logprob", "min_logprob", "repeated_state_count",
                    "tokens_since_progress", "thought_length", "action_length"):
            df[f"d_{col}"] = df[col].diff().fillna(0.0)
        df["logprob_slope3"] = (df["mean_logprob"].diff()
                                .rolling(3, min_periods=1).mean().fillna(0.0))
        # causal z-score: current logprob vs this run's PAST mean/std (no future)
        hist_mean = df["mean_logprob"].expanding(min_periods=2).mean().shift(1)
        hist_std = df["mean_logprob"].expanding(min_periods=2).std().shift(1)
        df["mean_logprob_z"] = (((df["mean_logprob"] - hist_mean) / hist_std)
                                .replace([float("inf"), float("-inf")], 0.0).fillna(0.0))
        # --- forward-looking target: collapse within H steps? (future-derived label) ---
        df["danger_zone"] = (df["steps_to_collapse"] <= 4).astype("int8")
        meta = runs.get(iid, {})
        es = meta.get("exit_status")
        pc = meta.get("patch_chars", len(preds.get(iid, {}).get("model_patch", "")))
        df["instance_id"] = iid
        df["repo"] = meta.get("repo", "")
        df["exit_status"] = es if es is not None else ""
        df["failure_mode"] = failure_mode(es or "", pc)
        df["is_last"] = df["step_index"] == df["step_index"].max()
        frames.append(df)
    if not frames:
        print("no feature parquets found")
        return
    data = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(OUT, index=False)

    print(f"wrote {OUT}  rows={len(data)}  trajectories={data.instance_id.nunique()}")
    print("\nfailure_mode (per trajectory):")
    print(data.drop_duplicates("instance_id").failure_mode.value_counts().to_string())
    nxt = data[~data.is_last]
    print(f"\nnext_step_error balance (non-terminal steps={len(nxt)}):")
    print(nxt.next_step_error.value_counts(dropna=False).to_string())
    print("\nloop_next balance:")
    print(nxt.loop_next.value_counts(dropna=False).to_string())
    print("\ndanger_zone balance (collapse within 4 steps):")
    print(data.danger_zone.value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()