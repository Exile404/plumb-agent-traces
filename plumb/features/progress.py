from __future__ import annotations

import hashlib
from typing import Sequence

from plumb.schemas import TrajectoryStep


def _state_hash(step: TrajectoryStep) -> str:
    payload = "|".join(sorted(step.files_touched)) + "::" + step.action
    return hashlib.sha1(payload.encode()).hexdigest()


def repeated_state(steps: Sequence[TrajectoryStep]) -> tuple[list[bool], list[int]]:
    """Whether the (files, action) state was seen earlier, and how many times before."""
    seen: dict[str, int] = {}
    flags, counts = [], []
    for step in steps:
        prior = seen.get(_state_hash(step), 0)
        flags.append(prior > 0)
        counts.append(prior)
        seen[_state_hash(step)] = prior + 1
    return flags, counts


def tokens_since_progress(steps: Sequence[TrajectoryStep]) -> list[int]:
    """Running token count since the last progress event (exit 0 or a new file)."""
    out, counter, seen_files = [], 0, set()
    for step in steps:
        counter += len(step.token_logprobs)
        out.append(counter)
        new_file = any(f not in seen_files for f in step.files_touched)
        seen_files.update(step.files_touched)
        if step.bash_exit_code == 0 or new_file:
            counter = 0
    return out


def file_progress(steps: Sequence[TrajectoryStep]) -> tuple[list[int], list[int]]:
    """Cumulative unique files touched, and repeats of the current step's files."""
    seen: dict[str, int] = {}
    unique_cum, same_repeats = [], []
    for step in steps:
        same_repeats.append(sum(seen.get(f, 0) for f in step.files_touched))
        for f in step.files_touched:
            seen[f] = seen.get(f, 0) + 1
        unique_cum.append(len(seen))
    return unique_cum, same_repeats