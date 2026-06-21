from __future__ import annotations

import math
from collections import Counter
from typing import Sequence


def tool_call_entropy(tool_names: Sequence[str | None]) -> float:
    """Shannon entropy (bits) over the tool-name distribution in the window."""
    names = [t for t in tool_names if t]
    if not names:
        return 0.0
    total = len(names)
    return -sum((c / total) * math.log2(c / total) for c in Counter(names).values())