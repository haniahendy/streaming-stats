"""P-squared quantile estimation (Jain & Chlamtac, 1985).

Maintains five markers and estimates a single quantile in O(1) space per
update. Accurate to a small error on smooth distributions without storing
any observations.
"""

from __future__ import annotations

from typing import List


class P2Quantile:
    def __init__(self, quantile: float = 0.5) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError("quantile must be in (0, 1)")
        self.p = quantile
        self._initial: List[float] = []
        self._q: List[float] = []   # marker heights
        self._n: List[int] = []     # marker positions
        self._np: List[float] = []  # desired positions
        self._dn: List[float] = []  # position increments

    def add(self, value: float) -> "P2Quantile":
        x = float(value)
        if len(self._initial) < 5:
            self._initial.append(x)
            if len(self._initial) == 5:
                self._bootstrap()
            return self

        q, n, np_, dn, p = self._q, self._n, self._np, self._dn, self.p
        # Find the cell k containing x and update extremes.
        if x < q[0]:
            q[0] = x
            k = 0
        elif x < q[1]:
            k = 0
        elif x < q[2]:
            k = 1
        elif x < q[3]:
            k = 2
        elif x <= q[4]:
            k = 3
        else:
            q[4] = x
            k = 3
        for i in range(k + 1, 5):
            n[i] += 1
        for i in range(5):
            np_[i] += dn[i]
        # Adjust marker heights where drift accumulates.
        for i in range(1, 4):
            d = np_[i] - n[i]
            if (d >= 1 and n[i + 1] - n[i] > 1) or (d <= -1 and n[i - 1] - n[i] < -1):
                step = 1 if d >= 1 else -1
                q_new = self._parabolic(i, step)
                q[i] = q_new if q[i - 1] < q_new < q[i + 1] else self._linear(i, step)
                n[i] += step
        return self

    def _bootstrap(self) -> None:
        self._q = sorted(self._initial)
        self._n = [1, 2, 3, 4, 5]
        p = self.p
        self._np = [1, 1 + 2 * p, 1 + 4 * p, 3 + 2 * p, 5]
        self._dn = [0, p / 2, p, (1 + p) / 2, 1]

    def _parabolic(self, i: int, step: int) -> float:
        q, n = self._q, self._n
        return q[i] + step / (n[i + 1] - n[i - 1]) * (
            (n[i] - n[i - 1] + step) * (q[i + 1] - q[i]) / (n[i + 1] - n[i])
            + (n[i + 1] - n[i] - step) * (q[i] - q[i - 1]) / (n[i] - n[i - 1]))

    def _linear(self, i: int, step: int) -> float:
        q, n = self._q, self._n
        return q[i] + step * (q[i + step] - q[i]) / (n[i + step] - n[i])

    @property
    def count(self) -> int:
        return self._n[4] if self._q else len(self._initial)

    @property
    def value(self) -> float:
        if not self._q:
            if not self._initial:
                raise ValueError("no values added")
            s = sorted(self._initial)
            return s[int(self.p * (len(s) - 1))]
        return self._q[2]
