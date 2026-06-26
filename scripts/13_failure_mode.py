"""Block E1 / H2: is the FAILURE MODE predictable from per-step behavioral
features? Two settings, both trajectory-level multiclass (overflow / stall /
submitted = wrong_fix+wrong_target; 'unknown' dropped as a data gap):
  full-run  : aggregate ALL steps incl. n_steps  -> post-hoc upper bound
              (n_steps is near-circular: a stall IS hitting the step limit)
  causal>=3 : aggregate only steps >= 3 before the end, NO n_steps -> the real
              "predict the mode in time to intervene" test (SPEC H2 lead time)
Stratified CV, macro-F1 vs the H2 bar of 0.5.

Run:  python scripts/13_failure_mode.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix

DATA = Path("data/dataset.parquet")
MODE = {"overflow": "overflow", "stall": "stall",
        "wrong_fix": "submitted", "wrong_target": "submitted"}

AGG = dict(
    n_steps=("step_index", "size"),
    logprob_mean=("mean_logprob", "mean"),
    logprob_last=("mean_logprob", "last"),
    min_logprob_min=("min_logprob", "min"),
    logprob_z_max=("mean_logprob_z", "max"),
    logprob_z_last=("mean_logprob_z", "last"),
    slope_mean=("logprob_slope3", "mean"),
    repeated_max=("repeated_state_count", "max"),
    repeated_rate=("repeated_state", "mean"),
    error_rate=("error_in_observation", "mean"),
    tokens_stuck_max=("tokens_since_progress", "max"),
    tokens_stuck_mean=("tokens_since_progress", "mean"),
    action_len_mean=("action_length", "mean"),
    thought_len_mean=("thought_length", "mean"),
    d_min_logprob_min=("d_min_logprob", "min"),
)


def run_features(d: pd.DataFrame, causal: bool) -> pd.DataFrame:
    if causal:
        d = d[d["steps_to_collapse"] >= 3]          # only what is visible >=3 steps early
    g = d.sort_values(["instance_id", "step_index"]).groupby("instance_id")
    run = g.agg(**AGG)
    if causal:
        run = run.drop(columns=["n_steps"])          # total length is not knowable mid-run
    run["mode"] = g["failure_mode"].first().map(MODE)
    return run.dropna(subset=["mode"])


def evaluate(run: pd.DataFrame, name: str) -> None:
    feats = [c for c in run.columns if c != "mode"]
    X = run[feats].astype(float).fillna(-1.0)
    y = run["mode"].to_numpy()
    classes = sorted(set(y))
    print(f"\n=== {name}  ({len(run)} runs: " +
          ", ".join(f"{c}={int((y == c).sum())}" for c in classes) + ") ===")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    oof = np.empty(len(y), dtype=object)
    imp = np.zeros(len(feats))
    for tr, te in skf.split(X, y):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                                 random_state=0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr])
        oof[te] = clf.predict(X.iloc[te])
        imp += clf.feature_importances_

    macro = f1_score(y, oof, average="macro")
    maj = pd.Series(y).value_counts().idxmax()
    base = f1_score(y, np.full(len(y), maj), average="macro")
    print(f"macro-F1 = {macro:.3f}   (majority baseline {base:.3f}, H2 bar 0.5)")
    print(classification_report(y, oof, digits=3, zero_division=0))
    print(f"confusion (rows=true, cols=pred) order {classes}:\n{confusion_matrix(y, oof, labels=classes)}")
    print("top features:", ", ".join(f for f, _ in sorted(zip(feats, imp), key=lambda t: -t[1])[:6]))


def main() -> None:
    d = pd.read_parquet(DATA).dropna(subset=["steps_to_collapse"])
    evaluate(run_features(d, causal=False), "full-run (post-hoc upper bound)")
    evaluate(run_features(d, causal=True), "causal: >=3 steps before failure, no n_steps")


if __name__ == "__main__":
    main()