from __future__ import annotations

from pathlib import Path

import pandas as pd

from plumb.schemas import RawSpan, TrajectoryOutcome, TrajectoryStep

DEFAULT_TRAJ_DIR = Path("data/trajectories")


def load_spans(jsonl_path: str | Path) -> list[RawSpan]:
    text = Path(jsonl_path).read_text(encoding="utf-8")
    return [RawSpan.model_validate_json(line) for line in text.splitlines() if line.strip()]


def spans_to_steps(spans: list[RawSpan]) -> tuple[str, list[TrajectoryStep]]:
    traj = next((s for s in spans if s.name == "trajectory"), None)
    trajectory_id = str(traj.attributes.get("plumb.trajectory_id", "")) if traj else ""

    step_index_by_span = {
        s.span_id: int(s.attributes["plumb.step_index"])
        for s in spans
        if s.name == "agent.step" and "plumb.step_index" in s.attributes
    }
    steps = {
        idx: TrajectoryStep(trajectory_id=trajectory_id, step_index=idx)
        for idx in step_index_by_span.values()
    }

    # Children attach to their agent.step parent. Last write wins per step,
    # which is fine for the 1 llm + 1 tool toy; generalised when we wire the real agent.
    for span in spans:
        idx = step_index_by_span.get(span.parent_span_id)
        if idx is None:
            continue
        step = steps[idx]
        attrs = span.attributes
        if span.name == "llm.call":
            content = str(attrs.get("gen_ai.response.content", ""))
            step.thought = content
            step.response_text = content
            step.token_logprobs = [float(x) for x in attrs.get("gen_ai.response.token_logprobs", [])]
            step.start_time_ns = span.start_time_ns
            step.end_time_ns = span.end_time_ns
        elif span.name == "tool.call":
            step.tool_name = attrs.get("plumb.tool_name")
            step.action = str(attrs.get("plumb.tool_command", ""))
            step.observation = str(attrs.get("plumb.tool_output", ""))
            exit_code = attrs.get("plumb.tool_exit_code")
            step.bash_exit_code = int(exit_code) if exit_code is not None else None

    return trajectory_id, [steps[i] for i in sorted(steps)]


def trajectory_outcome(trajectory_id: str, steps: list[TrajectoryStep]) -> TrajectoryOutcome:
    last_exit = steps[-1].bash_exit_code if steps else None
    failed = last_exit is not None and last_exit != 0
    return TrajectoryOutcome(trajectory_id=trajectory_id, failed=failed, n_steps=len(steps))


def steps_to_frame(steps: list[TrajectoryStep]) -> pd.DataFrame:
    return pd.DataFrame([s.model_dump() for s in steps])


def write_trajectory_parquet(jsonl_path: str | Path, out_dir: str | Path = DEFAULT_TRAJ_DIR) -> Path:
    trajectory_id, steps = spans_to_steps(load_spans(jsonl_path))
    out_path = Path(out_dir) / f"{trajectory_id}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    steps_to_frame(steps).to_parquet(out_path, index=False)
    return out_path