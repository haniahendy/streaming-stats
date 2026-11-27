"""Single-pass streaming statistics for unbounded numeric data.

Online algorithms that never store the stream: Welford moments,
exponential moving statistics, P-squared quantile estimation, and
reservoir sampling. For logs, sensor feeds, and metrics pipelines where
the data does not fit - or should not be kept - in memory.
"""

from .welford import OnlineMoments
from .ema import EMAMoments
from .p2quantile import P2Quantile
from .reservoir import ReservoirSample

__all__ = ["OnlineMoments", "EMAMoments", "P2Quantile", "ReservoirSample"]

__version__ = "0.1.0"
