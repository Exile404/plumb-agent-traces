"""H2 lead-time sweep: how does failure-mode foresight decay as we predict
EARLIER? For each lead L, aggregate only steps visible >= L before the end
(causal, no n_steps), classify overflow/stall/submitted, report macro-F1 and
per-class F1. Note: short runs drop out at large L (you can't call a 4-step
run 10 steps early), so the population shifts to long runs - read the per-class
lines and the printed counts, not just the aggregate. Writes figures/mode_leadtime.png.

Run:  python scripts/14_mode_leadtime.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

DATA = Path("data/dataset.parquet")
OUT = Path("figures/mode_leadtime.png")
MODE = {"overflow": "overflow", "stall": "stall",
        "wrong_fix": "submitted", "wrong_target": "submitted"}
CLASSES = ["overflow", "stall", "submitted"]
AGG = dict(
    logprob_mean=("mean_logprob", "mean"), logprob_last=("mean_logprob", "last"),
    min_logprob_min=("min_logprob", "min"), logprob_z_max=("mean_logprob_z", "max"),
    logprob_z_last=("mean_logprob_z", "last"), slope_mean=("logprob_slope3", "mean"),
    repeated_max=("repeated_state_count", "max"), repeated_rate=("repeated_state", "mean"),
    error_rate=("error_in_observation", "mean"), tokens_stuck_max=("tokens_since_progress", "max"),
    tokens_stuck_mean=("tokens_since_progress", "mean"), action_len_mean=("action_length", "mean"),
    thought_len_mean=("thought_length", "mean"), d_min_logprob_min=("d_min_logprob", "min"),
)


def run_features(d: pd.DataFrame, lead: int) -> pd.DataFrame:
    d = d[d["steps_to_collapse"] >= lead]
    g = d.sort_values(["instance_id", "step_index"]).groupby("instance_id")
    run = g.agg(**AGG)
    run["mode"] = g["failure_mode"].first().map(MODE)
    return run.dropna(subset=["mode"])


def macro_at(run: pd.DataFrame):
    feats = [c for c in run.columns if c != "mode"]
    X = run[feats].astype(float).fillna(-1.0)
    y = run["mode"].to_numpy()
    counts = {c: int((y == c).sum()) for c in CLASSES}
    present = [c for c in CLASSES if counts[c] > 0]
    mc = min(counts[c] for c in present) if present else 0
    if len(present) < 2 or mc < 3:
        return None, counts, len(run), {}
    oof = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(n_splits=min(5, mc), shuffle=True, random_state=0).split(X, y):
        clf = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                 min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                                 random_state=0, verbose=-1)
        clf.fit(X.iloc[tr], y[tr])
        oof[te] = clf.predict(X.iloc[te])
    per = {c: f1_score(y == c, oof == c, zero_division=0) for c in present}
    return f1_score(y, oof, average="macro"), counts, len(run), per


def main() -> None:
    d = pd.read_parquet(DATA).dropna(subset=["steps_to_collapse"])
    rows = []
    for L in [1, 2, 3, 4, 5, 7, 10]:
        macro, counts, n, per = macro_at(run_features(d, L))
        if macro is None:
            print(f"lead {L:>2}: skipped (a class < 3 runs)  counts={counts}  n={n}")
            continue
        rows.append((L, macro, n, per))
        print(f"lead {L:>2}: macro-F1 {macro:.3f}  n={n}  counts={counts}  "
              f"per={ {k: round(v, 2) for k, v in per.items()} }")

    Ls = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(Ls, [r[1] for r in rows], "-o", color="tab:blue", lw=2.2, label="macro-F1")
    for c, color in zip(CLASSES, ["tab:green", "tab:orange", "tab:red"]):
        ax.plot(Ls, [r[3].get(c, np.nan) for r in rows], "--o", color=color, ms=3, lw=1, label=f"{c} F1")
    ax.axhline(0.5, color="gray", ls=":", lw=1, label="H2 bar 0.5")
    ax.set_xlabel("lead time (agent steps before terminal failure)  ->  predicting earlier")
    ax.set_ylabel("F1"); ax.set_ylim(0, 1)
    ax.set_title("H2: failure-mode foresight vs how early you predict")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()