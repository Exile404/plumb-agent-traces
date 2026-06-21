from __future__ import annotations

from pydantic import BaseModel, Field


class RawSpan(BaseModel):
    """One captured OpenTelemetry span as written to the raw trace JSONL."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    kind: str = "INTERNAL"
    start_time_ns: int
    end_time_ns: int
    attributes: dict[str, object] = Field(default_factory=dict)


class TrajectoryStep(BaseModel):
    """One parsed agent step: thought, action, observation, and LLM signals."""

    trajectory_id: str
    step_index: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    tool_name: str | None = None
    response_text: str = ""
    token_logprobs: list[float] = Field(default_factory=list)
    files_touched: list[str] = Field(default_factory=list)
    bash_exit_code: int | None = None
    start_time_ns: int | None = None
    end_time_ns: int | None = None


class TrajectoryOutcome(BaseModel):
    """Terminal label for a whole trajectory."""

    trajectory_id: str
    failed: bool
    n_steps: int
    mast_class: str | None = None


class FeatureRow(BaseModel):
    """Per-step feature vector plus propagated labels. One row per agent step."""

    trajectory_id: str
    step_index: int

    tool_call_entropy: float = 0.0
    embedding_drift: float = 0.0
    recursion_depth: int = 0
    repeated_state: bool = False
    repeated_state_count: int = 0
    tokens_since_progress: int = 0
    mean_logprob: float = 0.0
    min_logprob: float = 0.0
    action_length: int = 0
    thought_length: int = 0
    error_in_observation: bool = False
    step_index_norm: float = 0.0
    unique_files_touched: int = 0
    same_file_repeats: int = 0
    bash_exit_code: int | None = None

    will_fail: bool | None = None
    steps_to_failure: int | None = None
    mast_class: str | None = None