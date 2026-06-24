"""Plumb intervention savings (Block E3): if we halt a failing run when its
collapse alarm first fires, how many agent steps do we avoid? Measured on the
all-failure corpus, so every halt is on a run that does fail (the benefit
ceiling). The false-halt cost in a mixed population is the precision line
(would-be successes wrongly halted). OOF signals-only probs, grouped CV.

Run:  python scripts/12_intervention.py   # writes figures/intervention_savings.png
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

DATA = Path("data/dataset.parquet")
OUT = Path("figures/intervention_savings.png")
TARGET = "danger_zone"
FEATURES = [
    "mean_logprob", "min_logprob", "error_in_observation", "repeated_state",
    "repeated_state_count", "tokens_since_progress", "action_length",
    "thought_length", "embedding_drift", "bash_exit_code",
    "d_mean_logprob", "d_min_logprob", "d_repeated_state_count",
    "d_tokens_since_progress", "d_thought_length", "d_action_length",
    "logprob_slope3", "mean_logprob_z",
]


def oof_probs(d: pd.DataFrame, feats: list[str]) -> np.ndarray:
    X = d[feats].astype(float).fillna(-1.0)
    y = d[TARGET].to_numpy()
    g = d["instance_id"].to_numpy()
    p = np.full(len(d), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, d.instance_id.nunique())).split(X, y, g):
        clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                 subsample=0.8, colsample_bytree=0.8, random_state=0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr])
        p[te] = clf.predict_proba(X.iloc[te])[:, 1]
    return p


def main() -> None:
    d = pd.read_parquet(DATA).dropna(subset=[TARGET]).copy()
    d[TARGET] = d[TARGET].astype(int)
    feats = [f for f in FEATURES if f in d.columns]
    d["p"] = oof_probs(d, feats)

    runs = d.groupby("instance_id")
    last = runs["step_index"].max()
    total_steps = int(runs.size().sum())
    n_runs = d.instance_id.nunique()

    rows = []
    for thr in np.round(np.linspace(0.2, 0.8, 13), 2):
        # first step each run crosses the alarm = realistic halt point
        fired = d[d["p"] >= thr].groupby("instance_id")["step_index"].min()
        saved = (last.loc[fired.index] - fired).clip(lower=0)   # steps avoided after halt
        flag = d["p"] >= thr
        nf = int(flag.sum())
        prec = int((flag & (d[TARGET] == 1)).sum()) / nf if nf else float("nan")
        rows.append({"thr": thr, "runs_fired_pct": 100 * len(fired) / n_runs,
                     "mean_saved_per_fired": float(saved.mean()) if len(saved) else 0.0,
                     "steps_saved_pct": 100 * float(saved.sum()) / total_steps,
                     "precision": prec})
    tab = pd.DataFrame(rows)

    print(f"corpus: {n_runs} failing runs, {total_steps} agent steps "
          f"(mean {total_steps / n_runs:.1f} steps/run)")
    print(tab.to_string(index=False, formatters={
        "thr": "{:.2f}".format, "runs_fired_pct": "{:.1f}".format,
        "mean_saved_per_fired": "{:.1f}".format, "steps_saved_pct": "{:.1f}".format,
        "precision": "{:.3f}".format}))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(tab["thr"], tab["steps_saved_pct"], "-o", color="tab:green",
            label="% of all agent steps saved")
    ax.set_xlabel("alarm threshold")
    ax.set_ylabel("% agent steps saved", color="tab:green")
    ax.tick_params(axis="y", labelcolor="tab:green")
    ax.set_ylim(0, max(60.0, tab["steps_saved_pct"].max() * 1.15))
    ax2 = ax.twinx()
    ax2.plot(tab["thr"], tab["precision"], "-s", color="tab:purple",
             label="precision (halt safety)")
    ax2.axhline(d[TARGET].mean(), color="tab:purple", ls=":", lw=1,
                label=f"base rate {d[TARGET].mean():.2f}")
    ax2.set_ylabel("precision", color="tab:purple")
    ax2.tick_params(axis="y", labelcolor="tab:purple"); ax2.set_ylim(0, 1)
    ax.set_title("Plumb intervention savings: steps avoided by halting failing runs at the alarm")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()