"""Command line entry point: streaming summary of a numeric column."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from .ema import EMAMoments
from .p2quantile import P2Quantile
from .welford import OnlineMoments


def _numbers(path: str, column):
    fh = sys.stdin if path == "-" else open(path, newline="", encoding="utf-8")
    with fh:
        if column is None:
            for line in fh:
                line = line.strip()
                if line:
                    yield float(line)
        else:
            reader = csv.DictReader(fh)
            if column not in (reader.fieldnames or []):
                raise SystemExit(f"missing column {column!r} in {path}")
            for record in reader:
                yield float(record[column])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="streamstats",
        description="One-pass streaming statistics over a numeric column.")
    parser.add_argument("input", help="CSV file, or '-' / a file of one number per line")
    parser.add_argument("--column", default=None, help="CSV column to read")
    parser.add_argument("--ema", type=float, default=None, metavar="ALPHA",
                        help="also compute an exponential moving average")
    parser.add_argument("--quantiles", default="0.5,0.9,0.99",
                        help="comma-separated P^2 quantiles")
    args = parser.parse_args(argv)

    moments = OnlineMoments()
    ema = EMAMoments(args.ema) if args.ema else None
    trackers = [(q, P2Quantile(q)) for q in
                (float(x) for x in args.quantiles.split(","))]
    for value in _numbers(args.input, args.column):
        moments.add(value)
        if ema:
            ema.add(value)
        for _, tracker in trackers:
            tracker.add(value)

    out = {
        "count": moments.count,
        "mean": round(moments.mean, 6),
        "stddev": round(moments.stddev, 6),
        "min": moments.min,
        "max": moments.max,
        "quantiles": {str(q): round(t.value, 6) for q, t in trackers},
    }
    if ema:
        out["ema"] = {"alpha": ema.alpha, "mean": round(ema.mean, 6),
                      "stddev": round(ema.stddev, 6)}
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
