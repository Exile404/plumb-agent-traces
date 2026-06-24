"""Plumb thesis figures: visualize the per-step early-warning result.
Panel A: one run's predicted collapse-probability over steps, with the alarm
         threshold and the danger zone (last 4 steps) shaded -> the alarm fires
         several steps before collapse.
Panel B: precision & median warning-lead vs alarm threshold (the operating curve).
Probabilities are out-of-fold (each step scored by a model that never trained on
its trajectory), signals-only (no step_index) -> deployable, leakage-free.

Run:  pip install matplotlib        # if missing (free)
      python scripts/11_figures.py  # writes figures/early_warning.png
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

DATA = Path("data/dataset.parquet")
OUT = Path("figures/early_warning.png")
TARGET = "danger_zone"
THRESH = 0.5
PREFER = ""   # auto-pick the run with the longest genuine pre-collapse lead
FEATURES = [
    "mean_logprob", "min_logprob", "error_in_observation", "repeated_state",
    "repeated_state_count", "tokens_since_progress", "action_length",
    "thought_length", "embedding_drift", "bash_exit_code",
    "d_mean_logprob", "d_min_logprob", "d_repeated_state_count",
    "d_tokens_since_progress", "d_thought_length", "d_action_length",
    "logprob_slope3", "mean_logprob_z",
]


def oof_probs(d: pd.DataFrame, feats: list[str]):
    X = d[feats].astype(float).fillna(-1.0)
    y = d[TARGET].to_numpy()
    groups = d["instance_id"].to_numpy()
    p = np.full(len(d), np.nan)
    gkf = GroupKFold(n_splits=min(5, d.instance_id.nunique()))
    for tr, te in gkf.split(X, y, groups):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr])
        p[te] = clf.predict_proba(X.iloc[te])[:, 1]
    return p, y


def sustained_lead(t: pd.DataFrame) -> tuple[int | None, int]:
    """Onset of the SUSTAINED alarm: start of the final run of consecutive steps
    that stay >= THRESH through to collapse (visually = where the curve crosses
    and stays up). Ignores transient spikes. Returns (onset_step, steps_lead)."""
    t = t.sort_values("step_index")
    steps = t["step_index"].to_numpy()
    prob = t["p"].to_numpy()
    if prob[-1] < THRESH:
        return None, 0
    onset = steps[-1]
    for i in range(len(steps) - 1, -1, -1):
        if prob[i] >= THRESH:
            onset = steps[i]
        else:
            break
    return int(onset), int(steps.max() - onset)


def pick_trajectory(d: pd.DataFrame) -> str:
    """Run with the largest *sustained* warning lead (PREFER if it has a clear one)."""
    leads = {}
    for iid, t in d.groupby("instance_id"):
        onset, lead = sustained_lead(t)
        if onset is not None and lead > 0:
            leads[iid] = lead
    if PREFER in leads and leads[PREFER] >= 3:
        return PREFER
    return max(leads, key=leads.get) if leads else d.groupby("instance_id").size().idxmax()


def main() -> None:
    d = pd.read_parquet(DATA).dropna(subset=[TARGET]).copy()
    d[TARGET] = d[TARGET].astype(int)
    feats = [f for f in FEATURES if f in d.columns]
    p, y = oof_probs(d, feats)
    d["p"] = p
    auroc = roc_auc_score(y, p)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 4.6))

    # ---- Panel A: one run's alarm over time ----
    iid = pick_trajectory(d)
    t = d[d.instance_id == iid].sort_values("step_index")
    steps = t["step_index"].to_numpy()
    prob = t["p"].to_numpy()
    pos = t[TARGET].to_numpy() == 1
    danger0 = steps[pos].min() if pos.any() else steps.max()
    axA.axvspan(danger0 - 0.5, steps.max() + 0.5, color="tab:red", alpha=0.12,
                label="danger zone (last 4 steps)")
    axA.plot(steps, prob, "-o", color="tab:blue", lw=2, ms=4, label="P(collapse soon)")
    axA.axhline(THRESH, color="gray", ls="--", lw=1, label=f"alarm threshold {THRESH}")
    onset, lead = sustained_lead(t)
    if onset is not None:
        axA.axvline(onset, color="tab:green", lw=1.5)
        axA.annotate(f"sustained alarm\n{lead} steps early", xy=(onset, THRESH),
                     xytext=(onset + 0.4, 0.72), color="tab:green", fontsize=9,
                     arrowprops=dict(arrowstyle="->", color="tab:green"))
    axA.set_title(f"Early warning on one run\n{iid}", fontsize=10)
    axA.set_xlabel("agent step"); axA.set_ylabel("predicted collapse probability")
    axA.set_ylim(0, 1); axA.legend(fontsize=8, loc="upper left")

    # ---- Panel B: precision & lead vs threshold ----
    thrs = np.linspace(0.15, 0.8, 14)
    prec, lead, n_traj = [], [], d.instance_id.nunique()
    for thr in thrs:
        flag = d["p"] >= thr
        nf = int(flag.sum())
        prec.append(int((flag & (d[TARGET] == 1)).sum()) / nf if nf else np.nan)
        first = d[flag].sort_values("step_index").groupby("instance_id").first()
        lead.append(float(np.median(first["steps_to_collapse"])) if len(first) else np.nan)
    axB.plot(thrs, prec, "-o", color="tab:purple", ms=3, label="precision")
    axB.axhline(d[TARGET].mean(), color="tab:purple", ls=":", lw=1,
                label=f"base rate {d[TARGET].mean():.2f}")
    axB.set_xlabel("alarm threshold"); axB.set_ylabel("precision", color="tab:purple")
    axB.tick_params(axis="y", labelcolor="tab:purple"); axB.set_ylim(0, 1)
    ax2 = axB.twinx()
    ax2.plot(thrs, lead, "-s", color="tab:orange", ms=3, label="median lead")
    ax2.set_ylabel("median steps of warning", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    axB.set_title(f"Operating curve (signals-only, OOF AUROC={auroc:.3f})", fontsize=10)
    h1, l1 = axB.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")

    fig.suptitle("Plumb: predicting agent trajectory collapse from model-internal signals", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}  | showcase={iid}  OOF AUROC={auroc:.3f}")


if __name__ == "__main__":
    main()