from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import tiktoken

from plumb.features.drift import embedding_drift
from plumb.features.entropy import tool_call_entropy
from plumb.features.progress import file_progress, repeated_state, tokens_since_progress
from plumb.parse import load_spans, spans_to_steps, trajectory_outcome
from plumb.schemas import FeatureRow, TrajectoryStep

DEFAULT_FEATURES_DIR = Path("data/features")
_ERROR_RE = re.compile(
    r"traceback|error|exception|fail|assert|cannot|not found|no such|denied", re.IGNORECASE
)

try:
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def _ntok(text: str) -> int:
    if not text:
        return 0
    return len(_ENCODING.encode(text)) if _ENCODING is not None else len(text.split())


def steps_to_features(steps: list[TrajectoryStep], window: int = 5) -> list[FeatureRow]:
    n = len(steps)
    failed = trajectory_outcome(steps[0].trajectory_id, steps).failed if steps else False
    drift = embedding_drift([s.observation for s in steps])
    rep_flags, rep_counts = repeated_state(steps)
    tsp = tokens_since_progress(steps)
    unique_cum, same_repeats = file_progress(steps)

    rows: list[FeatureRow] = []
    for i, step in enumerate(steps):
        window_tools = [s.tool_name for s in steps[max(0, i - window + 1): i + 1]]
        lps = step.token_logprobs
        rows.append(
            FeatureRow(
                trajectory_id=step.trajectory_id,
                step_index=step.step_index,
                tool_call_entropy=tool_call_entropy(window_tools),
                embedding_drift=drift[i],
                recursion_depth=0,  # stub until nested reasoning depth is captured
                repeated_state=rep_flags[i],
                repeated_state_count=rep_counts[i],
                tokens_since_progress=tsp[i],
                mean_logprob=sum(lps) / len(lps) if lps else 0.0,
                min_logprob=min(lps) if lps else 0.0,
                action_length=_ntok(step.action),
                thought_length=_ntok(step.thought),
                error_in_observation=bool(_ERROR_RE.search(step.observation)),
                step_index_norm=step.step_index / (n - 1) if n > 1 else 0.0,
                unique_files_touched=unique_cum[i],
                same_file_repeats=same_repeats[i],
                bash_exit_code=step.bash_exit_code,
                will_fail=failed,
                steps_to_failure=(n - 1 - i) if failed else None,
                mast_class=None,
            )
        )
    return rows


def features_to_frame(rows: list[FeatureRow]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump() for r in rows])


def extract_features_from_trace(
    jsonl_path: str | Path, out_dir: str | Path = DEFAULT_FEATURES_DIR
) -> Path:
    trajectory_id, steps = spans_to_steps(load_spans(jsonl_path))
    out_path = Path(out_dir) / f"{trajectory_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_to_frame(steps_to_features(steps)).to_parquet(out_path, index=False)
    return out_path