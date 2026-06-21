"""Capture one toy trajectory to data/raw_traces/<run_id>.jsonl via the local vLLM."""
from __future__ import annotations

import uuid

from openai import OpenAI

from plumb.tracing import init_tracing

RUN_ID = "toy-" + uuid.uuid4().hex[:8]
BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen-coder"
PROMPTS = [
    "Name the one python file to inspect for a failing import. One short line.",
    "Give a one-line bash command to run the tests.",
]


def main() -> None:
    tracer, provider, out_path = init_tracing(RUN_ID)
    client = OpenAI(base_url=BASE_URL, api_key="not-needed")
    with tracer.start_as_current_span("trajectory") as traj:
        traj.set_attribute("plumb.trajectory_id", RUN_ID)
        for step, prompt in enumerate(PROMPTS):
            with tracer.start_as_current_span("agent.step") as step_span:
                step_span.set_attribute("plumb.step_index", step)
                with tracer.start_as_current_span("llm.call") as llm_span:
                    resp = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=48,
                        temperature=0,
                        logprobs=True,
                        top_logprobs=1,
                    )
                    choice = resp.choices[0]
                    content = choice.message.content or ""
                    token_logprobs = (
                        [t.logprob for t in choice.logprobs.content]
                        if choice.logprobs and choice.logprobs.content
                        else []
                    )
                    llm_span.set_attribute("gen_ai.request.model", MODEL)
                    llm_span.set_attribute("gen_ai.prompt", prompt)
                    llm_span.set_attribute("gen_ai.response.content", content)
                    llm_span.set_attribute("gen_ai.response.token_logprobs", token_logprobs)
                    llm_span.set_attribute(
                        "gen_ai.usage.completion_tokens", resp.usage.completion_tokens
                    )
                with tracer.start_as_current_span("tool.call") as tool_span:
                    tool_span.set_attribute("plumb.tool_name", "bash")
                    tool_span.set_attribute("plumb.tool_command", "pytest -q")
                    tool_span.set_attribute("plumb.tool_exit_code", 1 if step == 0 else 0)
                    tool_span.set_attribute(
                        "plumb.tool_output",
                        "E ImportError: cannot import name foo" if step == 0 else "2 passed",
                    )
    provider.shutdown()
    print("wrote", out_path)


if __name__ == "__main__":
    main()