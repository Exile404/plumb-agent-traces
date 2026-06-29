# Plumb

**Predictive failure detection for long-horizon LLM coding agents.**

Plumb instruments an autonomous coding agent, reads cheap per-step signals from its execution trace, and forecasts task failure while the run is still in progress, early enough to intervene by restarting, replanning, or escalating to a stronger model. The signals come from data the agent already produces: per-token log-probabilities and behavioural cues such as repeated states, stalled progress, and error observations. A lightweight gradient-boosted model turns them into a calibrated, real-time estimate of how close a run is to collapse and in what manner it is likely to fail.

The project is a research prototype developed at Deakin University's Applied Artificial Intelligence Institute (A2I2), supervised by Prof. Rajesh Vasa.

## How it works

Plumb is a four-stage pipeline from a running agent to a failure forecast.

1. **Capture.** Each model call and tool call is recorded as an OpenTelemetry span (`plumb/tracing/`) to `data/raw_traces/<run>.jsonl`, including per-token log-probabilities.
2. **Reconstruct.** `plumb/parse.py` reassembles the spans into a per-step trajectory at `data/trajectories/<id>.parquet`.
3. **Featurize.** `plumb/features/` derives fourteen per-step features at `data/features/<id>.parquet`: model-internal confidence (mean and minimum log-probability and their trends) alongside behavioural signals (repeated state, tokens since progress, error-in-observation, action and reasoning length, and more).
4. **Model.** Gradient-boosted classifiers (LightGBM), trained with cross-validation grouped by trajectory and by repository, forecast imminent collapse, the eventual failure mode, and the compute recoverable by halting, using only causal features available at prediction time.

The reference agent is Mini-SWE-agent running against SWE-bench-Live, with Qwen2.5-Coder-7B served locally through vLLM. The capture and feature layers are agent- and model-agnostic.

## Findings

On a corpus of 87 trajectories (2,696 steps), with trajectory-grouped cross-validation:

- Imminent collapse is predicted at **AUROC 0.78 from real-time signals alone** (0.84 with step position), against a position-only baseline of 0.665. The strongest single predictor is the agent's confidence rising against its own recent baseline shortly before it derails.
- The **manner** of failure (context overflow, stall, or wrong submission) is predictable at least three steps ahead at **macro-F1 0.79**.
- The predictor transfers to **entirely held-out repositories at AUROC 0.79** across 56 repositories, indicating it reads agent behaviour rather than task specifics.
- Halting failing runs at the first alarm would recover an estimated **30 to 50 percent** of the agent steps otherwise spent failing.

Generated figures live in `figures/`: `early_warning.png`, `intervention_savings.png`, and `mode_leadtime.png`.

## Repository layout

```
plumb/        package: schemas, OpenTelemetry tracing, trajectory parsing, feature extraction
scripts/      numbered pipeline: serve model, run agent, capture, label, build dataset, train, figures
tests/        hermetic pipeline tests
data/         raw_traces, trajectories, features, datasets, predictions
figures/      generated result figures
notebooks/    analysis
```

## Setup

Python 3.12, in an in-project virtual environment:

```bash
python -m venv plumb-venv && source plumb-venv/bin/activate
pip install -e .
```

Serve the model in its own terminal and leave it running:

```bash
bash scripts/01_start_vllm.sh
```

## Quick start

Capture a trajectory and run it through the pipeline in a second terminal:

```bash
source plumb-venv/bin/activate
python scripts/05_run_mini_agent.py    # run the agent through the capture path (needs the server)
python scripts/03_parse_trajectory.py  # spans -> trajectory parquet
python scripts/04_extract_features.py  # trajectory -> features parquet
pytest -q
```

## Reproducing the analysis

The full research pipeline, from rollout to figures, requires the vLLM server and Docker:

```bash
python scripts/06_run_batch.py --sample 80 --max-files 2 --max-lines 60 --wall-time 900 --rmi
python scripts/07_label.py            # labels from the SWE-bench-Live evaluator
python scripts/08_build_dataset.py    # per-step targets and features
python scripts/09_train.py            # collapse predictor and position ablation
python scripts/11_figures.py          # -> figures/early_warning.png
python scripts/12_intervention.py     # -> figures/intervention_savings.png
python scripts/13_failure_mode.py     # failure-mode classifier
python scripts/14_mode_leadtime.py    # -> figures/mode_leadtime.png
python scripts/16_cross_repo.py       # cross-repository generalization
```

A free, mixed-outcome track on easier tasks, where the agent sometimes succeeds, is available through `scripts/17_run_pilot.py` and `scripts/18_train_pilot.py`. For modelling only, on the committed dataset, run scripts `08` through `16` after `pip install lightgbm scikit-learn matplotlib`.

## Stack

Qwen2.5-Coder-7B-Instruct-AWQ served on vLLM (OpenAI-compatible API with log-probabilities), OpenTelemetry GenAI tracing to JSONL, Parquet with pandas, LightGBM and scikit-learn for modelling, and matplotlib for figures. Python 3.12.

## Conventions

No em-dashes, result-first prose, typed public functions, Pydantic schemas, and filesystem paths through pathlib.
