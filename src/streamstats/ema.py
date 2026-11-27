"""Exponentially weighted moving average and variance."""

from __future__ import annotations

import math


class EMAMoments:
    """EMA and EM variance with a fixed smoothing factor alpha in (0, 1).

    alpha near 1 reacts fast; alpha near 0 is smooth. The variance uses
    the standard exponentially weighted moment recursion.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.count = 0
        self._mean = 0.0
        self._var = 0.0

    def add(self, value: float) -> "EMAMoments":
        x = float(value)
        self.count += 1
        if self.count == 1:
            self._mean = x
            self._var = 0.0
            return self
        delta = x - self._mean
        self._mean += self.alpha * delta
        self._var = (1.0 - self.alpha) * (self._var + self.alpha * delta * delta)
        return self

    @property
    def mean(self) -> float:
        if self.count == 0:
            raise ValueError("no values added")
        return self._mean

    @property
    def stddev(self) -> float:
        if self.count == 0:
            raise ValueError("no values added")
        return math.sqrt(self._var)
