from __future__ import annotations

from typing import Any

from minisweagent.agents.default import DefaultAgent
from opentelemetry.trace import Tracer


class TracedEnvironment:
    """Wraps a Mini-SWE-agent environment, emitting one tool.call span per execute."""

    def __init__(self, env: Any, tracer: Tracer) -> None:
        self._env = env
        self._tracer = tracer
        self.config = env.config

    def execute(self, action: dict, cwd: str = "") -> dict[str, Any]:
        with self._tracer.start_as_current_span("tool.call") as span:
            span.set_attribute("plumb.tool_name", "bash")
            span.set_attribute("plumb.tool_command", str(action.get("command", "")))
            output = self._env.execute(action, cwd)
            span.set_attribute("plumb.tool_exit_code", int(output.get("returncode", -1)))
            span.set_attribute("plumb.tool_output", str(output.get("output", ""))[:8000])
            return output

    def get_template_vars(self, **kwargs: Any) -> dict[str, Any]:
        return self._env.get_template_vars(**kwargs)

    def serialize(self) -> dict:
        return self._env.serialize()


class TracedAgent(DefaultAgent):
    """DefaultAgent instrumented for Plumb. Emits agent.step and llm.call spans and
    wraps the environment for tool.call spans. Pair with a parent 'trajectory' span
    opened by the run script."""

    def __init__(self, model: Any, env: Any, *, tracer: Tracer, **kwargs: Any) -> None:
        super().__init__(model, TracedEnvironment(env, tracer), **kwargs)
        self._tracer = tracer

    def step(self) -> list[dict]:
        with self._tracer.start_as_current_span("agent.step") as span:
            span.set_attribute("plumb.step_index", self.n_calls)
            return super().step()

    def query(self) -> dict:
        with self._tracer.start_as_current_span("llm.call") as span:
            message = super().query()
            response = message.get("extra", {}).get("response") or {}
            choice = (response.get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content") or message.get("content") or ""
            logprobs: list[float] = []
            entry = choice.get("logprobs")
            if isinstance(entry, dict) and entry.get("content"):
                logprobs = [t["logprob"] for t in entry["content"] if t.get("logprob") is not None]
            span.set_attribute("gen_ai.request.model", str(self.model.config.model_name))
            span.set_attribute("gen_ai.response.content", str(content)[:8000])
            span.set_attribute("gen_ai.response.token_logprobs", logprobs)
            return message