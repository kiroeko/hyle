#!/usr/bin/env python3
"""Unit tests for the strict common nanobench adapter."""

from __future__ import annotations

import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from hyle_perf.model import ValidationError  # noqa: E402
from hyle_perf.nanobench import (  # noqa: E402
    parse_nanobench,
    require_consistent_row_sets,
    select_pair,
)


ABSOLUTE = """
| ns/op | op/s | err% | total | demo absolute
|------:|-----:|-----:|------:|:----------
| 2.000 | 500,000,000 | 0.2% | 0.01 | `candidate`
| 2.200 | 454,545,455 | 0.3% | 0.01 | `peer`
"""

RELATIVE = """
| relative | ns/op | op/s | err% | total | demo relative
|---------:|------:|-----:|-----:|------:|:----------
| 100.0% | 1.000 | 1,000,000,000 | 0.1% | 0.01 | `candidate`
| 105.0% | 1.050 | 952,380,952 | 0.1% | 0.01 | `peer` | (Unstable with ~1.0 iters. Increase `minEpochIterations` to e.g. 10.)
"""

COMPARE = """
| relative | 95% CI | ns/op | op/s | err% | total | demo compare
|---------:|-------:|------:|-----:|-----:|------:|:------------
| 100.0% | | 14,676.65 | 68,135.45 | 4.2% | 0.14 | `candidate`
| 106.2% | 103.4% .. 111.3% | 13,757.49 | 72,687.70 | 3.8% | 0.12 | `peer`
"""


class NanobenchAdapterTest(unittest.TestCase):
    def test_parses_absolute_and_relative_tables(self):
        rows = parse_nanobench(ABSOLUTE + RELATIVE)
        candidate, peer = select_pair(rows, "candidate", "peer",
                                      title="demo relative")
        self.assertEqual(candidate.operations_per_second, 1_000_000_000.0)
        self.assertEqual(peer.operations_per_second, 952_380_952.0)

    def test_parses_paired_compare_table_by_header(self):
        rows = parse_nanobench(COMPARE)
        candidate, peer = select_pair(rows, "candidate", "peer",
                                      title="demo compare")
        self.assertEqual(candidate.operations_per_second, 68_135.45)
        self.assertEqual(peer.operations_per_second, 72_687.70)

    def test_header_without_operations_column_fails_closed(self):
        with self.assertRaisesRegex(ValidationError, "no op/s column"):
            parse_nanobench(COMPARE.replace("op/s", "items/s", 1))

    def test_pair_without_title_must_be_unambiguous(self):
        rows = parse_nanobench(ABSOLUTE + RELATIVE)
        with self.assertRaisesRegex(ValidationError, "ambiguous"):
            select_pair(rows, "candidate", "peer")

    def test_missing_pair_fails_closed(self):
        rows = parse_nanobench(ABSOLUTE)
        with self.assertRaisesRegex(ValidationError, "did not match"):
            select_pair(rows, "candidate", "absent")

    def test_duplicate_and_nonfinite_rows_fail_closed(self):
        with self.assertRaisesRegex(ValidationError, "duplicate"):
            parse_nanobench(ABSOLUTE + ABSOLUTE)
        with self.assertRaisesRegex(ValidationError, "finite and positive"):
            parse_nanobench(ABSOLUTE.replace("500,000,000", "nan"))

    def test_changed_row_set_fails_closed(self):
        first = parse_nanobench(ABSOLUTE)
        second = dict(first)
        del second[("demo absolute", "peer")]
        with self.assertRaisesRegex(ValidationError, "row set changed"):
            require_consistent_row_sets([first, second])


if __name__ == "__main__":
    unittest.main()
