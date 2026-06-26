"""Phase-2 capture: run OpenHands headless against local vLLM while recording
every LLM call (messages, content, per-token logprobs) to JSONL, forcing
logprobs=True so vLLM returns them. Runs in oh-venv (no plumb dependency).

Usage (oh-venv, vLLM up):
  export LLM_MODEL=openai/qwen-coder LLM_BASE_URL=http://localhost:8000/v1 \
         LLM_API_KEY=dummy RUNTIME=cli
  PLUMB_OH_TRACE=data/oh_traces/smoke.jsonl \
    python scripts/15_run_openhands.py -t "create hello.py that prints hi, then run it"
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

import litellm
from litellm.integrations.custom_logger import CustomLogger

TRACE = Path(os.environ.get("PLUMB_OH_TRACE", "data/oh_traces/run.jsonl")).resolve()
TRACE.parent.mkdir(parents=True, exist_ok=True)
TRACE.write_text("", encoding="utf-8")  # truncate per run


def _token_logprobs(resp) -> list[float]:
    try:
        lp = resp.choices[0].logprobs
        content = lp.get("content") if isinstance(lp, dict) else getattr(lp, "content", None)
        out = []
        for t in content or []:
            v = t.get("logprob") if isinstance(t, dict) else getattr(t, "logprob", None)
            if v is not None:
                out.append(float(v))
        return out
    except Exception:
        return []


class _Capture(CustomLogger):
    def _write(self, kwargs, response_obj):
        try:
            choice = response_obj.choices[0]
            tcs = getattr(choice.message, "tool_calls", None) or []
            rec = {
                "t_ns": time.time_ns(),
                "model": kwargs.get("model", ""),
                "messages": kwargs.get("messages", []),
                "content": getattr(choice.message, "content", "") or "",
                "tool_calls": [tc.model_dump() if hasattr(tc, "model_dump") else str(tc) for tc in tcs],
                "logprobs": _token_logprobs(response_obj),
            }
            with TRACE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:
            print(f"[plumb-capture] skipped a call: {e}", file=sys.stderr)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._write(kwargs, response_obj)


# force logprobs on every sync+async completion (patch at the litellm level so it
# applies even if OpenHands does not pass the param itself)
for _name in ("completion", "acompletion"):
    _orig = getattr(litellm, _name)

    def _wrap(orig=_orig):
        def inner(*a, **k):
            k.setdefault("logprobs", True)
            return orig(*a, **k)
        return inner

    setattr(litellm, _name, _wrap())

litellm.callbacks = [_Capture()]
print(f"[plumb-capture] active -> {TRACE}", file=sys.stderr)

# hand off to OpenHands' headless entrypoint with the remaining CLI args
sys.argv = ["openhands.core.main"] + sys.argv[1:]
runpy.run_module("openhands.core.main", run_name="__main__")