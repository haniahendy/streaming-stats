# streaming-stats

Single-pass statistics for unbounded numeric streams, in O(1) space per
estimator: Welford mean/variance, exponential moving moments, P-squared
quantiles, and reservoir sampling. For logs, sensor feeds, and metrics
pipelines where data should be processed, not stored.

Pure Python 3.10+, zero dependencies.

## Usage

```bash
seq 1 1000 | PYTHONPATH=src python3 -m streamstats.cli - --quantiles 0.5,0.9 --ema 0.1
```

```json
{
  "count": 1000, "mean": 500.5, "stddev": 288.67499, "min": 1.0, "max": 1000.0,
  "quantiles": {"0.5": 500.0, "0.9": 900.0},
  "ema": {"alpha": 0.1, "mean": 991.0, "stddev": 9.486833}
}
```

Input is one number per line, or a CSV with `--column name`. Library API:

- `OnlineMoments` - Welford count/mean/variance with min/max and a
  `merge` for combining parallel partial results.
- `EMAMoments(alpha)` - exponentially weighted mean/variance.
- `P2Quantile(p)` - Jain-Chlamtac P-squared single-quantile estimator.
- `ReservoirSample(k, seed)` - uniform fixed-size sample (Algorithm R),
  reproducible with a seed.

## Honest limits

- P-squared is an *estimator*: error is small on smooth distributions
  but it tracks one quantile each; for many quantiles or exact results,
  store the data or use a t-digest instead.
- Welford variance is numerically stable but not immune to catastrophic
  input (values near 1e308 overflow like any float pipeline).
- EMA assumes a fixed alpha chosen upfront; it is not adaptive to
  regime changes - that is what change-point detection is for.
- Reservoir sampling gives uniformity guarantees, not stratification;
  rare subpopulations can be missed entirely.
- `merge` combines moments only; quantile and reservoir estimators do
  not merge.

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

17 tests: Welford vs batch statistics, merge equivalence, EMA step math
and convergence, P-squared accuracy within tolerance on seeded data,
reservoir size/uniformity/determinism, input validation.
