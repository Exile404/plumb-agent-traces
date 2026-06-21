"""Parse the most recent raw trace JSONL into a trajectory parquet."""
from __future__ import annotations

import glob
import os

import pandas as pd

from plumb.parse import load_spans, spans_to_steps, trajectory_outcome, write_trajectory_parquet


def main() -> None:
    jsonl = max(glob.glob("data/raw_traces/*.jsonl"), key=os.path.getmtime)
    trajectory_id, steps = spans_to_steps(load_spans(jsonl))
    outcome = trajectory_outcome(trajectory_id, steps)
    out_path = write_trajectory_parquet(jsonl)
    frame = pd.read_parquet(out_path)
    print("source", jsonl)
    print(f"trajectory_id={trajectory_id} failed={outcome.failed} n_steps={outcome.n_steps}")
    print("wrote", out_path, frame.shape)
    print(frame[["step_index", "tool_name", "action", "bash_exit_code"]].to_string(index=False))
    print("token_logprobs per step:", [len(x) for x in frame["token_logprobs"]])


if __name__ == "__main__":
    main()