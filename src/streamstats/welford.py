"""Welford's online algorithm for mean and variance, with min/max."""

from __future__ import annotations

import math


class OnlineMoments:
    """Running count, mean, and population/sample variance in one pass."""

    def __init__(self) -> None:
        self.count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self.min = math.inf
        self.max = -math.inf

    def add(self, value: float) -> "OnlineMoments":
        x = float(value)
        self.count += 1
        delta = x - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (x - self._mean)
        self.min = min(self.min, x)
        self.max = max(self.max, x)
        return self

    def update(self, values) -> "OnlineMoments":
        for value in values:
            self.add(value)
        return self

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("no values added")
        return self._mean

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count == 0:
            raise ValueError("no values added")
        return self._m2 / self.count

    @property
    def sample_variance(self) -> float:
        if self.count < 2:
            raise ValueError("need at least 2 values")
        return self._m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)

    def merge(self, other: "OnlineMoments") -> "OnlineMoments":
        """Combine two independent partial results (parallel-friendly)."""
        if other.count == 0:
            return self
        if self.count == 0:
            self.count, self._mean, self._m2 = other.count, other._mean, other._m2
            self.min, self.max = other.min, other.max
            return self
        total = self.count + other.count
        delta = other._mean - self._mean
        self._m2 = (self._m2 + other._m2
                    + delta * delta * self.count * other.count / total)
        self._mean = (self._mean * self.count + other._mean * other.count) / total
        self.count = total
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        return self
