# Plumb

Predictive failure detection on long-horizon LLM agent trajectories.

A lightweight classifier reads per-step signals from an SWE agent's OpenTelemetry traces and forecasts task failure mid-run, early enough to intervene: restart, replan, escalate.

PhD prototype. Supervisor: Prof Rajesh Vasa, Deakin A2I2. The full thesis, hypotheses, stack rationale, and multi-week plan are kept in a separate local brief that is not part of this repository.

## Status

| Block | Scope | State |
|---|---|---|
| A | Quantized vLLM serving Qwen2.5-Coder-7B with logprobs (Blackwell sm_120) | Done |
| B | Capture harness: OTel spans to parquet to 14 features, pytest (Week-1 DoD) | Done |
| C | Mini-SWE-agent batch on SWE-bench-Live, real labels via the evaluator | Done |
| D | Collapse predictor (H1 reframed): AUROC 0.78 signals-only / 0.84 full, ~5-step lead | **Done** |
| E1 | Failure-mode head (H2): causal macro-F1 0.789 at ≥3-step lead | **Done** |
| E3 | Intervention savings: ~30–50% of agent steps recoverable | **Done** |
| E2 | H3: cross-repo generalization AUROC 0.789 (held-out repos); cross-agent budget-gated | Proxy done |
| F | Results writeup + 3 figures (seeded in local brief §19), deck, proposal v2 | In progress |

**← We are here:** the free track is essentially complete — Blocks A–E done on a zero-cost local-7B pipeline (87 trajectories / 2696 steps), with **four honest results + 3 figures**: (1) collapse prediction, (2) intervention savings, (3) failure-mode foresight, (4) cross-repo generalization. Everything left is **budget-gated** (literal pass-vs-fail H1, cross-*agent* H3 — both need a stronger model than the 16GB box can host) or the **writeup/deck (Block F)**.

### Done in detail
- vLLM serves `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` on an RTX 5060 Ti with per-token logprobs. Native sampler via `VLLM_USE_FLASHINFER_SAMPLER=0`, since the box has no CUDA toolkit for FlashInfer's runtime JIT.
- Capture pipeline, end to end and tested:
  - `plumb/tracing/` writes every LLM and tool span to `data/raw_traces/<run>.jsonl`.
  - `plumb/parse.py` reconstructs per-step trajectories to `data/trajectories/<id>.parquet`.
  - `plumb/features/` computes all 14 per-step features to `data/features/<id>.parquet`.
  - `tests/` cover capture round-trip and feature values, hermetic, green.
- Real Mini-SWE-agent (text-based model on local vLLM) runs through the same capture path via `plumb/agents/mini_wrapper.py` and `scripts/05_run_mini_agent.py`: a toy task returns `Submitted`, with genuine per-token logprobs and per-step bash commands reaching features.

### Results so far
Single-agent **all-failure** corpus: 87 trajectories / 2696 steps (7B via Mini-SWE-agent on SWE-bench-Live), trajectory-grouped CV.

1. **Collapse prediction (H1, reframed).** A LightGBM model predicts **imminent collapse** (within 4 steps of terminal failure) at **AUROC 0.78 from real-time internal signals alone** (0.84 with step position), beating a position-only baseline (0.665) with non-overlapping bars. Top signal: overconfidence relative to the run's own baseline (`mean_logprob_z`). Reframe of the literal H1 — pass-vs-fail needs a success class the 7B can't produce. Figure: `figures/early_warning.png`.
2. **Intervention savings (E3).** Halting doomed runs at the alarm recovers **~30% of agent steps at a balanced threshold, up to ~50%** aggressive. Figure: `figures/intervention_savings.png`.
3. **Failure-mode foresight (H2).** Which way a run fails (overflow / stall / submitted) is predictable **≥3 steps before terminal failure** at **macro-F1 0.789** from confidence-shape features (no run-length leakage), beating the 0.5 bar. **Stalls stay callable ~10 steps ahead** (F1 ~0.85). Figure: `figures/mode_leadtime.png`.
4. **Cross-repo generalization (H3 proxy).** Grouping CV by repo, collapse prediction holds at **AUROC 0.789 ± 0.010 on entirely held-out codebases** (56 repos) — the signal generalizes, it isn't memorizing repos. `scripts/16_cross_repo.py`. (Cross-*agent* H3 via OpenHands is budget-gated: capture built, but the 7B can't drive its tool API.)

Full writeup with caveats (ablations, leakage checks) in the local brief §19.

### Known gaps
- Single-agent all-failure data: no pass-vs-fail discrimination yet (deferred, budget-gated).
- `files_touched` not tracked, so unique-file features read 0. Placeholders: `recursion_depth` (stub), `embedding_drift` (hash embedding, swaps to bge-small).

## Stack

Qwen2.5-Coder-7B-Instruct-AWQ on vLLM (OpenAI API, logprobs). OpenTelemetry GenAI traces to JSONL. Parquet + pandas. LightGBM + scikit-learn (Blocks D/E1). matplotlib (figures). Python 3.12, in-project venv at `plumb-venv/`.

## Layout

```
plumb/        package: schemas, tracing, parse, features
scripts/      01 serve .. 05 mini agent  06 batch  07 label  08 dataset
              09 train  10 warning curve  11 figures  12 intervention
              13 failure-mode (H2)  14 mode lead-time
tests/        hermetic pipeline tests
data/         raw_traces, trajectories, features, dataset.parquet, predictions
figures/      early_warning.png  intervention_savings.png  mode_leadtime.png
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

## Reproduce the results

The research pipeline, from rollout to figures (needs the vLLM server + Docker):
```bash
python scripts/06_run_batch.py --sample 80 --max-files 2 --max-lines 60 --wall-time 900 --rmi
python scripts/07_label.py             # real labels from the SWE-bench-Live evaluator
python scripts/08_build_dataset.py     # per-step targets + features -> data/dataset.parquet
python scripts/09_train.py             # H1-reframe: collapse predictor + position ablation
python scripts/10_warning_curve.py     # precision / lead-time operating curve
python scripts/11_figures.py           # -> figures/early_warning.png
python scripts/12_intervention.py      # -> figures/intervention_savings.png
python scripts/13_failure_mode.py      # H2: failure-mode classifier (post-hoc + causal)
python scripts/14_mode_leadtime.py     # -> figures/mode_leadtime.png
```
Modeling only (no rollout, on the committed `data/dataset.parquet`): run `08`–`14` after `pip install lightgbm scikit-learn matplotlib`.

## Conventions

No em-dashes, result-first, typed public functions, Pydantic schemas, paths through pathlib, no markdown cells in notebooks.
