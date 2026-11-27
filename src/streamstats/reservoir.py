"""Reservoir sampling: uniform fixed-size sample of an unbounded stream."""

from __future__ import annotations

import random
from typing import List


class ReservoirSample:
    """Algorithm R: every stream item has equal probability k/n of being
    in the final sample, in O(k) space. Seeded for reproducibility."""

    def __init__(self, k: int, seed: int = 0) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.seen = 0
        self.sample: List[float] = []
        self._rng = random.Random(seed)

    def add(self, value) -> "ReservoirSample":
        self.seen += 1
        if len(self.sample) < self.k:
            self.sample.append(value)
        else:
            j = self._rng.randrange(self.seen)
            if j < self.k:
                self.sample[j] = value
        return self

    def update(self, values) -> "ReservoirSample":
        for value in values:
            self.add(value)
        return self
