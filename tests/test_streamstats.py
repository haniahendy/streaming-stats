import math
import random
import statistics
import unittest

from streamstats import EMAMoments, OnlineMoments, P2Quantile, ReservoirSample

DATA = [4.0, 7.0, 13.0, 16.0, 1.0, 9.0, 22.0, 5.5, 11.5, 3.0]


class TestOnlineMoments(unittest.TestCase):
    def test_matches_batch_statistics(self):
        m = OnlineMoments().update(DATA)
        self.assertEqual(m.count, len(DATA))
        self.assertAlmostEqual(m.mean, statistics.fmean(DATA))
        self.assertAlmostEqual(m.variance, statistics.pvariance(DATA))
        self.assertAlmostEqual(m.sample_variance, statistics.variance(DATA))
        self.assertEqual(m.min, min(DATA))
        self.assertEqual(m.max, max(DATA))

    def test_single_value(self):
        m = OnlineMoments().add(42.0)
        self.assertEqual(m.mean, 42.0)
        self.assertEqual(m.variance, 0.0)
        with self.assertRaises(ValueError):
            _ = m.sample_variance

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _ = OnlineMoments().mean

    def test_merge_matches_single_pass(self):
        a = OnlineMoments().update(DATA[:4])
        b = OnlineMoments().update(DATA[4:])
        a.merge(b)
        full = OnlineMoments().update(DATA)
        self.assertEqual(a.count, full.count)
        self.assertAlmostEqual(a.mean, full.mean)
        self.assertAlmostEqual(a.variance, full.variance)


class TestEMA(unittest.TestCase):
    def test_first_value_is_seed(self):
        ema = EMAMoments(0.5).add(10.0)
        self.assertEqual(ema.mean, 10.0)
        self.assertEqual(ema.stddev, 0.0)

    def test_converges_toward_constant_signal(self):
        ema = EMAMoments(0.2)
        for _ in range(200):
            ema.add(5.0)
        self.assertAlmostEqual(ema.mean, 5.0, places=6)
        self.assertAlmostEqual(ema.stddev, 0.0, places=6)

    def test_alpha_validation(self):
        with self.assertRaises(ValueError):
            EMAMoments(0.0)
        with self.assertRaises(ValueError):
            EMAMoments(1.0)

    def test_step_math(self):
        ema = EMAMoments(0.5)
        ema.add(0.0).add(10.0)
        # mean = 0 + 0.5*(10-0) = 5; var = 0.5*(0 + 0.5*100) = 25
        self.assertEqual(ema.mean, 5.0)
        self.assertAlmostEqual(ema.stddev, 5.0)


class TestP2Quantile(unittest.TestCase):
    def test_few_values_exact_sorted_pick(self):
        q = P2Quantile(0.5)
        q.add(3.0).add(1.0).add(2.0)
        self.assertEqual(q.value, 2.0)

    def test_median_close_on_sorted_range(self):
        values = list(range(1, 1001))
        q = P2Quantile(0.5).update if hasattr(P2Quantile, "update") else None
        tracker = P2Quantile(0.5)
        for v in values:
            tracker.add(v)
        self.assertAlmostEqual(tracker.value, 500.5, delta=5)

    def test_p90_close_on_seeded_random(self):
        rng = random.Random(42)
        values = [rng.gauss(100, 15) for _ in range(5000)]
        tracker = P2Quantile(0.9)
        for v in values:
            tracker.add(v)
        true_p90 = sorted(values)[int(0.9 * (len(values) - 1))]
        self.assertAlmostEqual(tracker.value, true_p90, delta=2.0)

    def test_constant_stream(self):
        tracker = P2Quantile(0.5)
        for _ in range(100):
            tracker.add(7.5)
        self.assertEqual(tracker.value, 7.5)

    def test_validation_and_empty(self):
        with self.assertRaises(ValueError):
            P2Quantile(1.5)
        with self.assertRaises(ValueError):
            _ = P2Quantile(0.5).value


class TestReservoir(unittest.TestCase):
    def test_size_and_prefix(self):
        r = ReservoirSample(5, seed=1).update(range(100))
        self.assertEqual(len(r.sample), 5)
        self.assertEqual(r.seen, 100)
        self.assertTrue(all(x in range(100) for x in r.sample))

    def test_short_stream_keeps_all(self):
        r = ReservoirSample(10, seed=1).update([1, 2, 3])
        self.assertEqual(r.sample, [1, 2, 3])

    def test_deterministic_with_seed(self):
        a = ReservoirSample(5, seed=7).update(range(50))
        b = ReservoirSample(5, seed=7).update(range(50))
        self.assertEqual(a.sample, b.sample)

    def test_validation(self):
        with self.assertRaises(ValueError):
            ReservoirSample(0)


if __name__ == "__main__":
    unittest.main()
