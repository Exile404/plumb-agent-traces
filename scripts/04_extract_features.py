"""Extract per-step features from the most recent raw trace into a features parquet."""
from __future__ import annotations

import glob
import os

import pandas as pd

from plumb.features.extract import extract_features_from_trace

FEATURE_COLS = [
    "tool_call_entropy", "embedding_drift", "recursion_depth", "repeated_state",
    "repeated_state_count", "tokens_since_progress", "mean_logprob", "min_logprob",
    "action_length", "thought_length", "error_in_observation", "step_index_norm",
    "unique_files_touched", "same_file_repeats", "bash_exit_code",
]


def main() -> None:
    jsonl = max(glob.glob("data/raw_traces/*.jsonl"), key=os.path.getmtime)
    out_path = extract_features_from_trace(jsonl)
    frame = pd.read_parquet(out_path)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("wrote", out_path, frame.shape)
    print(frame[["step_index", "mean_logprob", "min_logprob", "error_in_observation",
                 "tokens_since_progress", "repeated_state", "will_fail"]].to_string(index=False))
    print("15 feature cols present:", all(c in frame.columns for c in FEATURE_COLS))


if __name__ == "__main__":
    main()