"""Run the real Mini-SWE-agent on a toy task, capturing OTel spans for Plumb.

Requires the vLLM server up (scripts/01_start_vllm.sh). The agent executes
LLM-generated bash locally in data/agent_scratch; use a Docker environment for
untrusted or real SWE-bench tasks.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import yaml

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from minisweagent import package_dir
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

from plumb.agents import TracedAgent
from plumb.tracing import init_tracing

RUN_ID = "mini-" + uuid.uuid4().hex[:8]
TASK = (
    "Create a file named solution.txt in the current directory containing exactly the "
    "word: solved. Then show it with `cat solution.txt` to verify."
)


def main() -> None:
    cfg = yaml.safe_load((package_dir / "config" / "mini_textbased.yaml").read_text())
    agent_cfg = cfg["agent"]
    agent_cfg.pop("mode", None)
    agent_cfg["step_limit"] = 10
    model_cfg = cfg.get("model", {})
    model_kwargs = {
        **model_cfg.get("model_kwargs", {}),
        "api_base": "http://localhost:8000/v1",
        "api_key": "not-needed",
        "temperature": 0.0,
        "max_tokens": 1024,
        "logprobs": True,
        "top_logprobs": 1,
    }
    model = LitellmTextbasedModel(
        model_name="openai/qwen-coder",
        cost_tracking="ignore_errors",
        model_kwargs=model_kwargs,
        **{k: model_cfg[k] for k in ("observation_template", "format_error_template") if k in model_cfg},
    )
    scratch = Path("data/agent_scratch")
    scratch.mkdir(parents=True, exist_ok=True)
    env = LocalEnvironment(cwd=str(scratch.resolve()), env=cfg.get("environment", {}).get("env", {}))

    tracer, provider, out_path = init_tracing(RUN_ID)
    with tracer.start_as_current_span("trajectory") as traj:
        traj.set_attribute("plumb.trajectory_id", RUN_ID)
        agent = TracedAgent(model, env, tracer=tracer, **agent_cfg)
        result = agent.run(TASK)
    provider.shutdown()
    print("run_id:", RUN_ID, "| exit_status:", result.get("exit_status"))
    print("wrote:", out_path)


if __name__ == "__main__":
    main()