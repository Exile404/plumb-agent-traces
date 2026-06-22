# Plumb

Predictive failure detection on long-horizon LLM agent trajectories.

A lightweight classifier reads per-step signals from an SWE agent's OpenTelemetry traces and forecasts task failure mid-run, early enough to intervene: restart, replan, escalate.

PhD prototype. Supervisor: Prof Rajesh Vasa, Deakin A2I2. The full thesis, hypotheses, stack rationale, and multi-week plan are kept in a separate local brief that is not part of this repository.

## Status

| Block | Scope | State |
|---|---|---|
| A | Quantized vLLM serving Qwen2.5-Coder-7B with logprobs (Blackwell sm_120) | Done |
| B | Capture harness: OTel spans to parquet to 14 features, pytest (Week-1 DoD) | Done |
| C | Mini-SWE-agent wired into capture (done); batch on SWE-bench-Live (next) | In progress |
| D | Auto labels, train LightGBM, answer H1 (AUROC, lead-time, ablation) | Planned |
| E1 | MAST failure-mode head (H2) | Planned |
| E2 | OpenHands cross-agent transfer (H3) | Planned |
| E3 | Intervention savings figure | Planned |
| F | Results writeup, slide deck, proposal v2 | Planned |

### Done in detail
- vLLM serves `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` on an RTX 5060 Ti with per-token logprobs. Native sampler via `VLLM_USE_FLASHINFER_SAMPLER=0`, since the box has no CUDA toolkit for FlashInfer's runtime JIT.
- Capture pipeline, end to end and tested:
  - `plumb/tracing/` writes every LLM and tool span to `data/raw_traces/<run>.jsonl`.
  - `plumb/parse.py` reconstructs per-step trajectories to `data/trajectories/<id>.parquet`.
  - `plumb/features/` computes all 14 per-step features to `data/features/<id>.parquet`.
  - `tests/` cover capture round-trip and feature values, hermetic, green.
- Real Mini-SWE-agent (text-based model on local vLLM) runs through the same capture path via `plumb/agents/mini_wrapper.py` and `scripts/05_run_mini_agent.py`: a toy task returns `Submitted`, with genuine per-token logprobs and per-step bash commands reaching features.

### Remaining and known gaps
- Batch rollouts on SWE-bench-Live not started; real tasks need the Docker environment (the toy run uses LocalEnvironment).
- Labels use an exit-code proxy; real labels come from agent `exit_status` plus SWE-bench test verification. `files_touched` is not tracked yet, so unique-file features read 0.
- Placeholders: `recursion_depth` (stub), `embedding_drift` (cheap hash embedding, swaps to bge-small).
- No classifier yet.

## Stack

Qwen2.5-Coder-7B-Instruct-AWQ on vLLM (OpenAI API, logprobs). OpenTelemetry GenAI traces to JSONL. Parquet + pandas. LightGBM (Block D). Python 3.12, in-project venv at `plumb-venv/`.

## Layout

```
plumb/        package: schemas, tracing, parse, features
scripts/      01 serve  02 toy trace  03 parse  04 features
tests/        hermetic pipeline tests
data/         raw_traces, trajectories, features, labels, models
notebooks/    analysis (Block D onward)
```

## Quick start

Serve the model (own terminal, leave running):
```bash
source plumb-venv/bin/activate
bash scripts/01_start_vllm.sh
```

Capture a toy trajectory and run the pipeline (second terminal):
```bash
source plumb-venv/bin/activate
python scripts/02_smoke_trace.py       # toy trace to data/raw_traces/<run>.jsonl
python scripts/05_run_mini_agent.py    # or a real Mini-SWE-agent run (needs the server)
python scripts/03_parse_trajectory.py  # parses the newest trace to data/trajectories/<id>.parquet
python scripts/04_extract_features.py  # to data/features/<id>.parquet
pytest -q                              # 3 passed
```

## Conventions

No em-dashes, result-first, typed public functions, Pydantic schemas, paths through pathlib, no markdown cells in notebooks.
