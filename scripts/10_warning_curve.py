"""Plumb early-warning operating points: turn the danger_zone AUROC into a
deployment story. Signals-only (no step_index) danger model, out-of-fold
probabilities under trajectory-grouped CV, then per-step precision/recall +
lead-time (how many steps before collapse each run's FIRST alarm fires).

Run:  python scripts/10_warning_curve.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DATA = Path("data/dataset.parquet")
TARGET = "danger_zone"
# signals-only: causal real-time features, NO step_index (deployable mid-run)
FEATURES = [
    "mean_logprob", "min_logprob", "error_in_observation", "repeated_state",
    "repeated_state_count", "tokens_since_progress", "action_length",
    "thought_length", "embedding_drift", "bash_exit_code",
    "d_mean_logprob", "d_min_logprob", "d_repeated_state_count",
    "d_tokens_since_progress", "d_thought_length", "d_action_length",
    "logprob_slope3", "mean_logprob_z",
]


def main() -> None:
    d = pd.read_parquet(DATA).dropna(subset=[TARGET]).copy()
    d[TARGET] = d[TARGET].astype(int)
    feats = [f for f in FEATURES if f in d.columns]
    X = d[feats].astype(float).fillna(-1.0)
    y = d[TARGET].to_numpy()
    groups = d["instance_id"].to_numpy()

    # out-of-fold probabilities: each step scored by a model that never saw its run
    oof = np.full(len(d), np.nan)
    gkf = GroupKFold(n_splits=min(5, d.instance_id.nunique()))
    for tr, te in gkf.split(X, y, groups):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr])
        oof[te] = clf.predict_proba(X.iloc[te])[:, 1]
    d["p"] = oof
    print(f"danger_zone signals-only  OOF AUROC={roc_auc_score(y, oof):.3f}  "
          f"steps={len(d)}  trajectories={d.instance_id.nunique()}  base_rate={y.mean():.3f}")

    n_traj = d.instance_id.nunique()
    n_pos = int((d[TARGET] == 1).sum())
    rows = []
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        flag = d["p"] >= thr
        tp = int((flag & (d[TARGET] == 1)).sum())
        n_flag = int(flag.sum())
        prec = tp / n_flag if n_flag else float("nan")
        rec = tp / n_pos if n_pos else float("nan")
        # each run's FIRST alarm -> steps_to_collapse there = warning lead time
        first = d[flag].sort_values("step_index").groupby("instance_id").first()
        leads = first["steps_to_collapse"].to_numpy()
        med_lead = float(np.median(leads)) if len(leads) else float("nan")
        rows.append((thr, 100 * n_flag / len(d), prec, rec, 100 * len(first) / n_traj, med_lead))

    print(f"\n{'thr':>5}{'flag%':>8}{'prec':>8}{'recall':>8}{'traj%':>8}{'med_lead':>10}")
    for thr, fr, pr, rc, cv, ml in rows:
        print(f"{thr:>5.2f}{fr:>8.1f}{pr:>8.3f}{rc:>8.3f}{cv:>8.1f}{ml:>10.1f}")
    print("\nmed_lead = median steps-before-collapse of each run's FIRST alarm "
          "(higher = earlier warning); traj% = runs that fire at all.")


if __name__ == "__main__":
    main()