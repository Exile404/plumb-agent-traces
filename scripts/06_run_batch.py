"""Block C batch: real Mini-SWE-agent on SWE-bench-Live in Docker, captured
through the Plumb OTel pipeline (one trace per instance).

Prereqs: vLLM up (scripts/01_start_vllm.sh) + Docker running + `datasets` installed.
Smoke (1 instance):  python scripts/06_run_batch.py --slice 0:1
Each instance writes data/raw_traces/<instance_id>.jsonl and (parse on by default)
data/trajectories/<id>.parquet + data/features/<id>.parquet. Resume skips instances
whose raw trace already exists (--redo to force). Agent patches accumulate in
data/predictions/<split>.json for later SWE-bench-Live verification (Block D labels).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import traceback
from pathlib import Path

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
DATASET = "SWE-bench-Live/SWE-bench-Live"

PLUMB_TASK_GUIDANCE = """

# CRITICAL - how to actually fix this repository
The target repository is ALREADY checked out at /testbed (your working directory). Your fix MUST modify the repository's existing, version-tracked source files - not new files you create.
1. FIRST explore the real codebase with `ls`, `find`, and `grep` to locate the actual source file(s) that implement the buggy behavior. Do NOT write a standalone reproduction script and "fix" that - editing a new untracked file does NOT change the repository and produces an empty patch.
2. Make your minimal edit directly in the real source file(s) under /testbed.
3. Before submitting, run `git diff` and CONFIRM it is NON-EMPTY and shows your change to the real source. An empty `git diff` means you edited the wrong thing - go back and fix the actual tracked file.
Only run the submit command after you have verified a correct, non-empty `git diff`.
"""


def live_image_name(instance_id: str) -> str:
    """SWE-bench-Live DockerHub image: org starryzhang, __ -> _1776_, lowercased."""
    return f"starryzhang/sweb.eval.x86_64.{instance_id.replace('__', '_1776_').lower()}"


def ensure_image(image: str) -> None:
    """Pull with visible progress if not already local, keeping the multi-GB
    download out of DockerEnvironment's pull_timeout."""
    present = subprocess.run(
        ["docker", "image", "inspect", image],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if present:
        return
    print(f"    pulling {image} (first time, multi-GB)...", flush=True)
    subprocess.run(["docker", "pull", image], check=True)

def remove_image(image: str) -> None:
    subprocess.run(["docker", "rmi", "-f", image], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_model(cfg: dict, api_base: str, max_tokens: int) -> LitellmTextbasedModel:
    model_cfg = cfg.get("model", {})
    model_kwargs = {
        **model_cfg.get("model_kwargs", {}),  # drop_params: true
        "api_base": api_base,
        "api_key": "not-needed",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "logprobs": True,
        "top_logprobs": 1,
    }
    return LitellmTextbasedModel(
        model_name="openai/qwen-coder",
        cost_tracking="ignore_errors",
        model_kwargs=model_kwargs,
        **{k: model_cfg[k] for k in ("observation_template", "format_error_template") if k in model_cfg},
    )


def build_env(env_cfg: dict, instance: dict):
    env_cfg = copy.deepcopy(env_cfg)
    env_cfg.setdefault("environment_class", "docker")
    env_cfg["image"] = live_image_name(instance["instance_id"])
    return get_environment(env_cfg)


def update_preds(preds_path: Path, instance_id: str, model_name: str, patch: str) -> None:
    preds_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(preds_path.read_text()) if preds_path.exists() else {}
    data[instance_id] = {"model_name_or_path": model_name, "instance_id": instance_id, "model_patch": patch}
    preds_path.write_text(json.dumps(data, indent=2))
    
def update_runs(runs_path: Path, instance_id: str, info: dict) -> None:
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(runs_path.read_text()) if runs_path.exists() else {}
    data[instance_id] = info
    runs_path.write_text(json.dumps(data, indent=2))

def _distribution(instances: list[dict], key: str) -> str:
    from collections import Counter
    counts = Counter(str(x.get(key, "")) for x in instances)
    return ", ".join(f"{k or '<none>'}:{n}" for k, n in counts.most_common())


def _difficulty_dict(x: dict) -> dict:
    """Gold-patch size descriptor, e.g. {'files':1,'hunks':1,'lines':6}."""
    d = x.get("difficulty")
    if isinstance(d, dict):
        return d
    if isinstance(d, str):
        try:
            return json.loads(d.replace("'", '"'))
        except Exception:
            return {}
    return {}


def select_instances(instances: list[dict], *, difficulty: str, repos: str, max_files: int,
                     max_lines: int, sample: int, slice_spec: str) -> list[dict]:
    import re
    if repos:
        rx = re.compile(repos)
        instances = [x for x in instances if rx.search(str(x.get("repo", "")))]
    if difficulty:
        rx = re.compile(difficulty, re.IGNORECASE)
        instances = [x for x in instances if rx.search(str(x.get("difficulty", "")))]
    if max_files or max_lines:
        big = 1 << 30
        instances = [
            x for x in instances
            if (not max_files or _difficulty_dict(x).get("files", big) <= max_files)
            and (not max_lines or _difficulty_dict(x).get("lines", big) <= max_lines)
        ]
    instances = sorted(instances, key=lambda x: x["instance_id"])
    if not sample:
        start, stop = (int(x) if x else None for x in slice_spec.split(":"))
        return instances[start:stop]
    by_repo: dict[str, list[dict]] = {}                       # round-robin across repos for diversity
    for x in instances:
        by_repo.setdefault(str(x.get("repo", "")), []).append(x)
    order = sorted(by_repo)
    picked: list[dict] = []
    idx = 0
    while len(picked) < sample and any(by_repo[r] for r in order):
        pool = by_repo[order[idx % len(order)]]
        if pool:
            picked.append(pool.pop(0))
        idx += 1
    return picked


def run_instance(instance: dict, model, agent_cfg: dict, env_cfg: dict, traces_dir: Path) -> dict:
    iid = instance["instance_id"]
    ensure_image(live_image_name(iid))
    tracer, provider, out_path = init_tracing(iid, traces_dir)
    exit_status, submission = None, ""
    try:
        env = build_env(env_cfg, instance)
        try:
            with tracer.start_as_current_span("trajectory") as traj:
                traj.set_attribute("plumb.trajectory_id", iid)
                traj.set_attribute("plumb.instance_id", iid)
                traj.set_attribute("plumb.repo", str(instance.get("repo", "")))
                traj.set_attribute("plumb.base_commit", str(instance.get("base_commit", "")))
                agent = TracedAgent(model, env, tracer=tracer, **agent_cfg)
                try:
                    result = agent.run(instance["problem_statement"])
                    exit_status, submission = result.get("exit_status"), result.get("submission", "")
                except Exception as e:  # capture partial run (e.g. context overflow) as a failure
                    exit_status = type(e).__name__
                    traj.set_attribute("plumb.error", str(e)[:500])
        finally:
            getattr(env, "cleanup", lambda: None)()
    finally:
        provider.shutdown()
    return {"instance_id": iid, "exit_status": exit_status, "submission": submission, "trace": out_path}


def main() -> None:
    ap = argparse.ArgumentParser(description="Plumb Block C batch on SWE-bench-Live")
    ap.add_argument("--split", default="lite")
    ap.add_argument("--slice", default="0:1", help="start:stop over the sorted split, e.g. 0:1")
    ap.add_argument("--difficulty", default="", help="regex on the difficulty field (case-insensitive)")
    ap.add_argument("--max-files", type=int, default=0, help="keep tasks whose gold patch touches <= N files (0=off)")
    ap.add_argument("--max-lines", type=int, default=0, help="keep tasks whose gold patch changes <= N lines (0=off)")
    ap.add_argument("--repos", default="", help="regex on the repo field, e.g. 'pandas|requests'")
    ap.add_argument("--sample", type=int, default=0, help="pick N instances round-robin across repos (diversity); overrides --slice")
    ap.add_argument("--dry-run", action="store_true", help="print the selection (difficulty/repo spread) and exit, no runs")
    ap.add_argument("--step-limit", type=int, default=50)
    ap.add_argument("--wall-time", type=int, default=0, help="per-instance wall-clock cap, seconds (0=off)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--pull-timeout", type=int, default=1800)
    ap.add_argument("--api-base", default="http://localhost:8000/v1")
    ap.add_argument("--traces-dir", default="data/raw_traces")
    ap.add_argument("--no-parse", action="store_true", help="capture only; skip trajectory+feature parquet")
    ap.add_argument("--redo", action="store_true", help="rerun instances even if a trace exists")
    ap.add_argument("--rmi", action="store_true", help="docker rmi each image after use (bounds disk on large runs)")
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = yaml.safe_load(CONFIG.read_text())
    env_cfg = dict(cfg.get("environment", {}))
    env_cfg["pull_timeout"] = args.pull_timeout
    agent_cfg = dict(cfg.get("agent", {}))
    agent_cfg.pop("mode", None)
    agent_cfg["step_limit"] = args.step_limit
    if args.wall_time:
        agent_cfg["wall_time_limit_seconds"] = args.wall_time
    agent_cfg["instance_template"] = agent_cfg.get("instance_template", "") + PLUMB_TASK_GUIDANCE

    model = build_model(cfg, args.api_base, args.max_tokens)

    all_instances = sorted(load_dataset(DATASET, split=args.split), key=lambda x: x["instance_id"])
    print(f"[batch] split={args.split}: {len(all_instances)} instances | difficulty -> {_distribution(all_instances, 'difficulty')}")
    instances = select_instances(all_instances, difficulty=args.difficulty, repos=args.repos,
                                 max_files=args.max_files, max_lines=args.max_lines,
                                 sample=args.sample, slice_spec=args.slice)
    print(f"[batch] selected {len(instances)} | repos -> {_distribution(instances, 'repo')}"
          f" | difficulty -> {_distribution(instances, 'difficulty')}")
    if args.dry_run:
        for x in instances:
            print(f"  {x['instance_id']:48} [{x.get('difficulty', '')}]  {x.get('repo', '')}")
        return

    traces_dir = Path(args.traces_dir)
    preds_path = Path("data/predictions") / f"{args.split}.json"
    runs_path = Path("data/runs.json")
    print(f"[batch] step_limit={args.step_limit} max_tokens={args.max_tokens} wall_time={args.wall_time}")

    for i, instance in enumerate(instances, 1):
        iid = instance["instance_id"]
        trace = traces_dir / f"{iid}.jsonl"
        if trace.exists() and trace.stat().st_size > 0 and not args.redo:
            print(f"[{i}/{len(instances)}] skip (trace exists): {iid}")
            continue
        print(f"[{i}/{len(instances)}] {iid}")
        try:
            res = run_instance(instance, model, agent_cfg, env_cfg, traces_dir)
        except Exception as e:
            print(f"    ERROR {type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        update_preds(preds_path, iid, model.config.model_name, res["submission"])
        update_runs(runs_path, iid, {"exit_status": res["exit_status"],
                                "patch_chars": len(res["submission"]),
                                "repo": str(instance.get("repo", "")),
                                "difficulty": str(instance.get("difficulty", ""))})
        print(f"    exit_status={res['exit_status']} patch_chars={len(res['submission'])} -> {res['trace']}")
        if not args.no_parse:
            try:
                tp = write_trajectory_parquet(res["trace"])
                fp = extract_features_from_trace(res["trace"])
                print(f"    parsed -> {tp.name} | features -> {fp.name}")
            except Exception as e:
                print(f"    parse/features skipped ({type(e).__name__}: {e})")
        if args.rmi:
            remove_image(live_image_name(iid))


if __name__ == "__main__":
    main()