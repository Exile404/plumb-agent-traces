from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_DIM = 64


def cheap_embedding(text: str, dim: int = _DIM) -> list[float]:
    """Deterministic hashing embedding. Placeholder for bge-small until the
    embed extra is installed. L2-normalised token-hash count vector."""
    vec = [0.0] * dim
    for token in _TOKEN_RE.findall(text.lower()):
        idx = int.from_bytes(hashlib.sha1(token.encode()).digest()[:4], "big") % dim
        vec[idx] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def embedding_drift(observations: Sequence[str], k: int = 3) -> list[float]:
    """Per-step cosine distance between the current observation embedding and
    the rolling mean of the previous k observation embeddings."""
    embeddings = [cheap_embedding(o) for o in observations]
    drift: list[float] = []
    for i, emb in enumerate(embeddings):
        prev = embeddings[max(0, i - k):i]
        if not prev:
            drift.append(0.0)
            continue
        mean = [sum(col) / len(prev) for col in zip(*prev)]
        norm = math.sqrt(sum(v * v for v in mean))
        mean = [v / norm for v in mean] if norm else mean
        drift.append(1.0 - sum(x * y for x, y in zip(emb, mean)))
    return drift