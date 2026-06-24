"""Plumb per-step early-warning models (free, single-agent all-failure data).
Two targets, trajectory-grouped CV (no run leaks across train/test):
  next_step_error : will the agent's NEXT action error?     baseline = current error flag
  danger_zone     : will the run collapse within 4 steps?   baseline = step position
Features are causal (known at step t); d_* are within-run step-over-step changes.

Run:  python scripts/09_train.py     (needs: pip install lightgbm scikit-learn)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

DATA = Path("data/dataset.parquet")
FEATURES = [
    "step_index", "mean_logprob", "min_logprob", "error_in_observation",
    "repeated_state", "repeated_state_count", "tokens_since_progress",
    "action_length", "thought_length", "embedding_drift", "bash_exit_code",
    # trend / escalation features (within-run step-over-step change)
    "d_mean_logprob", "d_min_logprob", "d_repeated_state_count",
    "d_tokens_since_progress", "d_thought_length", "d_action_length",
    "logprob_slope3", "mean_logprob_z",
]


def evaluate(df: pd.DataFrame, target: str, baseline_col: str, drop: tuple[str, ...] = ()) -> None:
    feats = [f for f in FEATURES if f not in drop]
    d = df.dropna(subset=[target]).copy()
    d[target] = d[target].astype(int)
    X = d[feats].astype(float).fillna(-1.0)
    y = d[target].to_numpy()
    groups = d["instance_id"].to_numpy()
    base = d[baseline_col].astype(float).fillna(-1.0).to_numpy()

    n_splits = min(5, d.instance_id.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    aucs, aps, b_aucs, imp = [], [], [], np.zeros(len(feats))
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr])
        p = clf.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        aps.append(average_precision_score(y[te], p))
        b_aucs.append(roc_auc_score(y[te], base[te]))
        imp += clf.feature_importances_

    label = target + (f"  [drop {','.join(drop)}]" if drop else "")
    print(f"\n=== {label}  (rows={len(d)}, positive_rate={y.mean():.3f}) ===")
    print(f"  AUROC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  AP={np.mean(aps):.3f}  "
          f"(baseline '{baseline_col}' AUROC={np.mean(b_aucs):.3f})")
    print("  top features (LightGBM gain, summed):")
    for f, v in sorted(zip(feats, imp), key=lambda t: -t[1])[:8]:
        print(f"    {f:24} {v:8.0f}")


def main() -> None:
    df = pd.read_parquet(DATA)
    print(f"trajectories={df.instance_id.nunique()}  steps={len(df)}")
    evaluate(df[~df.is_last], "next_step_error", "error_in_observation")
    evaluate(df, "danger_zone", "step_index")
    evaluate(df, "danger_zone", "step_index", drop=("step_index",))  # ablation: no position


if __name__ == "__main__":
    main()
