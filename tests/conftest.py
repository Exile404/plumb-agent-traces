from __future__ import annotations

from pathlib import Path

import pytest

from plumb.schemas import RawSpan


def _span(name, span_id, parent, attrs, start=0, end=1):
    return RawSpan(
        trace_id="0" * 32, span_id=span_id, parent_span_id=parent,
        name=name, start_time_ns=start, end_time_ns=end, attributes=attrs,
    )


@pytest.fixture
def toy_trace_jsonl(tmp_path: Path) -> Path:
    """Deterministic two-step trace: step 0 errors (exit 1), step 1 passes (exit 0)."""
    spans = [
        _span("trajectory", "t0", None, {"plumb.trajectory_id": "toy-test"}),
        _span("agent.step", "s0", "t0", {"plumb.step_index": 0}),
        _span("llm.call", "l0", "s0", {
            "gen_ai.response.content": "inspect main.py",
            "gen_ai.response.token_logprobs": [-0.1, -0.2, -0.9],
        }),
        _span("tool.call", "c0", "s0", {
            "plumb.tool_name": "bash", "plumb.tool_command": "pytest -q",
            "plumb.tool_exit_code": 1,
            "plumb.tool_output": "E ImportError: cannot import name foo",
        }),
        _span("agent.step", "s1", "t0", {"plumb.step_index": 1}),
        _span("llm.call", "l1", "s1", {
            "gen_ai.response.content": "run the tests",
            "gen_ai.response.token_logprobs": [-0.3, -0.4],
        }),
        _span("tool.call", "c1", "s1", {
            "plumb.tool_name": "bash", "plumb.tool_command": "pytest -q",
            "plumb.tool_exit_code": 0, "plumb.tool_output": "2 passed",
        }),
    ]
    path = tmp_path / "toy-test.jsonl"
    path.write_text("\n".join(s.model_dump_json() for s in spans), encoding="utf-8")
    return path