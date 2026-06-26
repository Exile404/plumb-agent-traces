"""Train the mixed-corpus pilot: predict EVENTUAL run failure from the trajectory so
far. The all-failure SWE-bench corpus could not test this; here some runs succeed, so
AUROC measures real success/failure separation, not proximity to a known end.

Reads data/pilot_features/*.parquet (labeled by scripts/17), rebuilds the same causal
features as scripts/08, and reports (trajectory-grouped CV):
  - outcome AUROC/AP + Brier (calibration) from causal signals
  - surface-only ablation (drop model-internal logprob features)
  - position-only baseline (step index)
  - early-prediction AUROC (first 5 steps only)
  - cross-model AUROC (GroupKFold by model) if >1 model present
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

FEATURES_DIR = Path("data/pilot_features")

SIGNAL = [
    "mean_logprob", "min_logprob", "mean_logprob_z", "logprob_slope3",
    "repeated_state", "repeated_state_count", "tokens_since_progress",
    "error_in_observation", "action_length", "thought_length",
    "embedding_drift", "bash_exit_code", "tool_call_entropy",
    "d_mean_logprob", "d_min_logprob", "d_repeated_state_count",
    "d_tokens_since_progress", "d_thought_length", "d_action_length",
]
LOGPROB = ["mean_logprob", "min_logprob", "mean_logprob_z", "logprob_slope3",
           "d_mean_logprob", "d_min_logprob"]
POSITION = ["step_index", "step_index_norm"]


def add_causal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("step_index").reset_index(drop=True)
    for col in ("mean_logprob", "min_logprob", "repeated_state_count",
                "tokens_since_progress", "thought_length", "action_length"):
        df[f"d_{col}"] = df[col].diff().fillna(0.0)
    df["logprob_slope3"] = (df["mean_logprob"].diff()
                            .rolling(3, min_periods=1).mean().fillna(0.0))
    hist_mean = df["mean_logprob"].expanding(min_periods=2).mean().shift(1)
    hist_std = df["mean_logprob"].expanding(min_periods=2).std().shift(1)
    df["mean_logprob_z"] = (((df["mean_logprob"] - hist_mean) / hist_std)
                            .replace([np.inf, -np.inf], 0.0).fillna(0.0))
    return df


def load() -> pd.DataFrame:
    frames = []
    for fp in sorted(FEATURES_DIR.glob("*.parquet")):
        df = pd.read_parquet(fp)
        if "will_fail" not in df.columns or len(df) == 0:
            continue
        df = add_causal(df)
        df["trajectory_id"] = fp.stem
        df["model"] = df["model"].iloc[0] if "model" in df.columns else "unknown"
        df["run_failed"] = df["will_fail"].astype(int)
        frames.append(df)
    if not frames:
        raise SystemExit("no pilot feature parquets in data/pilot_features/")
    return pd.concat(frames, ignore_index=True)


def lgbm() -> LGBMClassifier:
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                          subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
                          random_state=0, n_jobs=-1, verbose=-1)


def oof(df: pd.DataFrame, feats: list[str], group: str) -> np.ndarray:
    X, y, g = df[feats].to_numpy(), df["run_failed"].to_numpy(), df[group].to_numpy()
    pred = np.zeros(len(df))
    gkf = GroupKFold(n_splits=min(5, df[group].nunique()))
    for tr, te in gkf.split(X, y, g):
        pred[te] = lgbm().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return pred


def boot_auroc(y, p, groups, n=1000, seed=0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_g = {gg: np.where(groups == gg)[0] for gg in uniq}
    aucs = []
    for _ in range(n):
        ii = np.concatenate([idx_by_g[gg] for gg in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) == 2:
            aucs.append(roc_auc_score(y[ii], p[ii]))
    return tuple(np.percentile(aucs, [2.5, 97.5])) if aucs else (float("nan"), float("nan"))


def report(name: str, df: pd.DataFrame, y, p) -> None:
    lo, hi = boot_auroc(y, p, df["trajectory_id"].to_numpy())
    print(f"  {name:32} AUROC {roc_auc_score(y, p):.3f} [{lo:.3f},{hi:.3f}]  "
          f"AP {average_precision_score(y, p):.3f}  Brier {brier_score_loss(y, p):.3f}")


def main() -> None:
    df = load()
    runs = df.drop_duplicates("trajectory_id")
    print(f"[pilot] runs={len(runs)} steps={len(df)}  "
          f"fail={int(runs.run_failed.sum())} success={int((1 - runs.run_failed).sum())}")
    print("per model (size, fail-rate):")
    print(runs.groupby("model").run_failed.agg(["size", "mean"]).to_string())
    if runs.run_failed.nunique() < 2:
        print("\n[pilot] corpus is single-outcome; need both successes and failures to model.")
        return

    y = df["run_failed"].to_numpy()
    print("\nPredicting eventual run failure (trajectory-grouped CV):")
    report("causal signals (no position)", df, y, oof(df, SIGNAL, "trajectory_id"))
    report("surface-only (drop logprobs)", df, y,
           oof(df, [f for f in SIGNAL if f not in LOGPROB], "trajectory_id"))
    report("position-only baseline", df, y, oof(df, POSITION, "trajectory_id"))

    p_sig = oof(df, SIGNAL, "trajectory_id")
    early = (df["step_index"] < 5).to_numpy()
    if early.sum() > 10 and len(np.unique(y[early])) == 2:
        print(f"\n  early (first 5 steps) AUROC {roc_auc_score(y[early], p_sig[early]):.3f} "
              f"on {int(early.sum())} steps")

    if df["model"].nunique() > 1:
        print("\nCross-model (GroupKFold by model, held-out model per fold):")
        report("causal signals, held-out model", df, y, oof(df, SIGNAL, "model"))


if __name__ == "__main__":
    main()
