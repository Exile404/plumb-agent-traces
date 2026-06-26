"""Mixed-corpus pilot: Mini-SWE-agent on easy MBPP tasks through the Plumb capture
path, so the model SOMETIMES SUCCEEDS. This gives the success/failure spread the
all-failure SWE-bench-Live corpus lacks, so we can test whether per-step signals
predict EVENTUAL OUTCOME, not just proximity to a known end.

Each MBPP task becomes a tiny working dir: solution.py (empty, the agent fills it)
and test_task.py (pytest file from the task asserts). The agent edits solution.py
and runs the tests in a LocalEnvironment (no Docker). After it stops, WE run pytest:
pass -> success, else failure. That verdict is the ground-truth label on the parquet.

Serve one model at a time, then run with a matching --model-tag:
  # terminal 1: bash scripts/01_start_vllm.sh   (served name e.g. qwen-coder)
  python scripts/17_run_pilot.py --served-name qwen-coder --model-tag qwen7b  --n 50
  # restart vLLM on a weaker model, then:
  python scripts/17_run_pilot.py --served-name qwen-coder --model-tag qwen1_5b --n 50

SANDBOX NOTE: LocalEnvironment runs the agent's bash on THIS machine (scoped to each
task dir, 30s timeout). Fine for toy tasks on your box; do not point it at anything
you care about.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pandas as pd
import yaml

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

from minisweagent import package_dir
from minisweagent.environments import get_environment
from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

from plumb.agents import TracedAgent
from plumb.features.extract import extract_features_from_trace
from plumb.parse import write_trajectory_parquet
from plumb.tracing import init_tracing

CONFIG = package_dir / "config" / "benchmarks" / "swebench_backticks.yaml"
TASKS_DIR = Path("data/pilot_tasks")
TRACES_DIR = Path("data/pilot_traces")
FEATURES_DIR = Path("data/pilot_features")
RUNS = Path("data/pilot_runs.json")

INSTANCE_TEMPLATE = """\
You are given a small Python programming task. Implement the solution in `solution.py`
in your working directory so the provided tests pass.

## Task
{{task}}

## Working rules
- The working directory already has `solution.py` (empty) and `test_task.py` (the tests).
- Put your implementation in `solution.py`. Define every function the tests call at module level.
- Write files non-interactively with a heredoc, for example:
  ```mswea_bash_command
  cat > solution.py <<'PY'
  def my_func(x):
      return x
  PY
  ```
- Run the tests with `python -m pytest -q test_task.py` and iterate until they pass.
- Never use interactive editors (vi/nano).

## Submitting
Once `python -m pytest -q test_task.py` shows all tests pass, submit with EXACTLY:
```mswea_bash_command
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && echo done
```
"""


def build_test_file(task: dict) -> str:
    setup = (task.get("test_setup_code") or "").strip()
    asserts = [a for a in (task.get("test_list") or []) if a.strip()]
    body = "\n".join("    " + a for a in asserts) or "    pass"
    head = "from solution import *\n" + (setup + "\n" if setup else "")
    return f"{head}\n\ndef test_solution():\n{body}\n"


def problem_statement(task: dict) -> str:
    text = (task.get("text") or task.get("prompt") or "").strip()
    tests = "\n".join(task.get("test_list") or [])
    return f"{text}\n\nThe solution must satisfy these tests:\n{tests}"


def make_task_dir(task: dict, model_tag: str) -> Path:
    d = TASKS_DIR / model_tag / f"mbpp_{task['task_id']}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "solution.py").write_text("")
    (d / "test_task.py").write_text(build_test_file(task))
    return d


def run_tests(task_dir: Path) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_task.py"],
        cwd=task_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=120,
    )
    return proc.returncode == 0


def build_model(cfg: dict, served_name: str, api_base: str, max_tokens: int) -> LitellmTextbasedModel:
    model_cfg = cfg.get("model", {})
    model_kwargs = {
        **model_cfg.get("model_kwargs", {}),
        "api_base": api_base, "api_key": "not-needed",
        "temperature": 0.0, "max_tokens": max_tokens,
        "logprobs": True, "top_logprobs": 1,
    }
    return LitellmTextbasedModel(
        model_name=f"openai/{served_name}",
        cost_tracking="ignore_errors",
        model_kwargs=model_kwargs,
        **{k: model_cfg[k] for k in ("observation_template", "format_error_template") if k in model_cfg},
    )


def relabel(fp: Path, resolved: bool, model_tag: str) -> int:
    df = pd.read_parquet(fp)
    n = len(df)
    df["will_fail"] = not resolved
    df["steps_to_failure"] = list(range(n - 1, -1, -1)) if not resolved else pd.NA
    df["model"] = model_tag
    df.to_parquet(fp, index=False)
    return n


def run_one(task: dict, model, agent_cfg: dict, model_tag: str) -> dict:
    tid = f"pilot-{model_tag}-mbpp_{task['task_id']}"
    task_dir = make_task_dir(task, model_tag)
    tracer, provider, trace_path = init_tracing(tid, TRACES_DIR)
    exit_status = None
    try:
        env = get_environment({
            "environment_class": "local",
            "cwd": str(task_dir.resolve()),
            "timeout": 30,
            "env": {"PAGER": "cat", "PIP_PROGRESS_BAR": "off", "TQDM_DISABLE": "1"},
        })
        with tracer.start_as_current_span("trajectory") as traj:
            traj.set_attribute("plumb.trajectory_id", tid)
            traj.set_attribute("plumb.instance_id", tid)
            traj.set_attribute("plumb.repo", "mbpp")
            traj.set_attribute("plumb.model", model_tag)
            agent = TracedAgent(model, env, tracer=tracer, **agent_cfg)
            try:
                exit_status = agent.run(problem_statement(task)).get("exit_status")
            except Exception as e:
                exit_status = type(e).__name__
                traj.set_attribute("plumb.error", str(e)[:500])
    finally:
        provider.shutdown()
    resolved = False
    try:
        resolved = run_tests(task_dir)
    except Exception as e:
        print(f"    pytest error: {type(e).__name__}: {e}")
    return {"tid": tid, "trace": trace_path, "exit_status": exit_status, "resolved": resolved}


def update_runs(tid: str, info: dict) -> None:
    RUNS.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(RUNS.read_text()) if RUNS.exists() else {}
    data[tid] = info
    RUNS.write_text(json.dumps(data, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Plumb mixed-corpus pilot on MBPP")
    ap.add_argument("--served-name", default="qwen-coder", help="vLLM served model id (name in scripts/01)")
    ap.add_argument("--model-tag", required=True, help="short label stored on rows, e.g. qwen7b")
    ap.add_argument("--n", type=int, default=50, help="number of MBPP tasks")
    ap.add_argument("--start", type=int, default=0, help="offset into the sorted split")
    ap.add_argument("--split", default="test")
    ap.add_argument("--step-limit", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--redo", action="store_true")
    args = ap.parse_args()

    from datasets import load_dataset

    for d in (TASKS_DIR, TRACES_DIR, FEATURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(CONFIG.read_text())
    agent_cfg = {
        "system_template": cfg["agent"]["system_template"],
        "instance_template": INSTANCE_TEMPLATE,
        "step_limit": args.step_limit,
        "cost_limit": 1e9,
        "max_consecutive_format_errors": 3,
    }
    model = build_model(cfg, args.served_name, args.api_base, args.max_tokens)

    ds = sorted(load_dataset("google-research-datasets/mbpp", "full", split=args.split),
                key=lambda x: x["task_id"])
    tasks = ds[args.start: args.start + args.n]
    print(f"[pilot] model_tag={args.model_tag} served={args.served_name} "
          f"tasks={len(tasks)} step_limit={args.step_limit}")

    n_pass = n_run = 0
    for i, task in enumerate(tasks, 1):
        tid = f"pilot-{args.model_tag}-mbpp_{task['task_id']}"
        trace = TRACES_DIR / f"{tid}.jsonl"
        if trace.exists() and trace.stat().st_size > 0 and not args.redo:
            print(f"[{i}/{len(tasks)}] skip (exists): {tid}")
            continue
        print(f"[{i}/{len(tasks)}] {tid}")
        try:
            res = run_one(task, model, agent_cfg, args.model_tag)
        except Exception as e:
            print(f"    ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        n_run += 1
        n_pass += int(res["resolved"])
        try:
            write_trajectory_parquet(res["trace"])
            fp = extract_features_from_trace(res["trace"], out_dir=FEATURES_DIR)
            n_steps = relabel(fp, res["resolved"], args.model_tag)
        except Exception as e:
            print(f"    parse/features skipped ({type(e).__name__}: {e})")
            n_steps = 0
        update_runs(tid, {"model": args.model_tag, "resolved": res["resolved"],
                          "exit_status": res["exit_status"], "n_steps": n_steps,
                          "task_id": task["task_id"]})
        print(f"    resolved={res['resolved']} exit={res['exit_status']} steps={n_steps}")
    print(f"\n[pilot] {args.model_tag}: pass={n_pass}/{n_run} attempted "
          f"(success rate {n_pass / max(1, n_run):.0%})")


if __name__ == "__main__":
    main()
