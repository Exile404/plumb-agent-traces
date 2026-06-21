from __future__ import annotations

import math
from pathlib import Path

from plumb.features.extract import steps_to_features
from plumb.parse import load_spans, spans_to_steps


def test_pipeline_shapes_and_values(toy_trace_jsonl: Path):
    trajectory_id, steps = spans_to_steps(load_spans(toy_trace_jsonl))
    assert trajectory_id == "toy-test"
    assert [s.step_index for s in steps] == [0, 1]
    assert [s.bash_exit_code for s in steps] == [1, 0]
    assert [len(s.token_logprobs) for s in steps] == [3, 2]

    rows = steps_to_features(steps)
    assert len(rows) == 2
    r0, r1 = rows

    assert r0.error_in_observation is True
    assert r1.error_in_observation is False
    assert math.isclose(r0.mean_logprob, -1.2 / 3, rel_tol=1e-9)
    assert math.isclose(r0.min_logprob, -0.9, rel_tol=1e-9)
    assert r0.tokens_since_progress == 3
    assert r1.tokens_since_progress == 5  # 3 + 2, reset happens after step 1
    assert r0.repeated_state is False
    assert r1.repeated_state is True      # same (files, action) state as step 0
    assert r1.repeated_state_count == 1
    assert r0.step_index_norm == 0.0
    assert r1.step_index_norm == 1.0


def test_labels_propagate(toy_trace_jsonl: Path):
    _, steps = spans_to_steps(load_spans(toy_trace_jsonl))
    rows = steps_to_features(steps)
    assert all(r.will_fail is False for r in rows)        # last exit 0 = pass
    assert all(r.steps_to_failure is None for r in rows)