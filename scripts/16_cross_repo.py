"""H3 proxy (cross-repo generalization): does the collapse signal transfer to
UNSEEN codebases? Same danger_zone signals-only model as 09, but CV grouped by
REPO (held-out repos in test) instead of by trajectory. If AUROC holds, the
signal generalizes across task domains, not memorizing one repo. Free; uses the
existing mini dataset (OpenHands cross-agent H3 is budget-gated, see SPEC 19).

Run:  python scripts/16_cross_repo.py
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
    # repo group from instance_id: "owner__repo-1234" -> "owner__repo"
    d["repo_group"] = d["instance_id"].str.rsplit("-", n=1).str[0]
    feats = [f for f in FEATURES if f in d.columns]
    X = d[feats].astype(float).fillna(-1.0)
    y = d[TARGET].to_numpy()
    groups = d["repo_group"].to_numpy()
    n_repos = d.repo_group.nunique()
    print(f"repos={n_repos}  trajectories={d.instance_id.nunique()}  steps={len(d)}  pos_rate={y.mean():.3f}")

    gkf = GroupKFold(n_splits=min(5, n_repos))
    aucs = []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X.iloc[te])[:, 1]))

    print(f"\ndanger_zone signals-only, HELD-OUT REPOS:  AUROC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  (folds={len(aucs)})")
    print("vs trajectory-grouped CV ~0.78 (scripts/09). Holding here => signal generalizes across codebases.")


if __name__ == "__main__":
    main()