from __future__ import annotations

from pathlib import Path

from plumb.parse import load_spans, spans_to_steps
from plumb.tracing import init_tracing


def test_tracer_writes_parseable_jsonl(tmp_path: Path):
    tracer, provider, out_path = init_tracing("cap-test", traces_dir=tmp_path)
    with tracer.start_as_current_span("trajectory") as traj:
        traj.set_attribute("plumb.trajectory_id", "cap-test")
        with tracer.start_as_current_span("agent.step") as step:
            step.set_attribute("plumb.step_index", 0)
            with tracer.start_as_current_span("llm.call") as llm:
                llm.set_attribute("gen_ai.response.content", "ok")
                llm.set_attribute("gen_ai.response.token_logprobs", [-0.5, -0.6])
            with tracer.start_as_current_span("tool.call") as tool:
                tool.set_attribute("plumb.tool_name", "bash")
                tool.set_attribute("plumb.tool_command", "ls")
                tool.set_attribute("plumb.tool_exit_code", 0)
                tool.set_attribute("plumb.tool_output", "file.py")
    provider.shutdown()

    assert out_path.exists()
    spans = load_spans(out_path)
    assert len(spans) == 4  # trajectory + step + llm + tool
    trajectory_id, steps = spans_to_steps(spans)
    assert trajectory_id == "cap-test"
    assert len(steps) == 1
    assert steps[0].token_logprobs == [-0.5, -0.6]
    assert steps[0].tool_name == "bash"
    assert steps[0].bash_exit_code == 0