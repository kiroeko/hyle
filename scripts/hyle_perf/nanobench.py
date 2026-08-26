"""Strict nanobench adapter for the HYLE performance instrument.

This module only translates one child-process table into raw metric values. It
does not choose margins, estimators, stopping rules, or verdicts; those belong
to the immutable experiment contract and the common statistical core.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Mapping

from hyle_perf.model import ValidationError


_BACKTICK = "`"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


@dataclass(frozen=True)
class NanobenchRow:
    title: str
    variant: str
    operations_per_second: float


def parse_nanobench(text: str) -> dict[tuple[str, str], NanobenchRow]:
    """Parse one nanobench process and reject ambiguous or invalid rows."""
    if not isinstance(text, str):
        raise ValidationError("nanobench output must be text")
    rows: dict[tuple[str, str], NanobenchRow] = {}
    title = None
    operations_column = None
    for original_line in text.splitlines():
        line = _ANSI_ESCAPE.sub("", original_line)
        if "|" not in line:
            continue
        columns = [column.strip() for column in line.split("|")]
        if len(columns) < 6:
            continue
        if columns[1] == "relative":
            title = columns[-1]
            operations_column = 3
            continue
        if columns[1] == "ns/op":
            title = columns[-1]
            operations_column = 2
            continue
        variant_column = next(
            (column for column in columns[5:] if _BACKTICK in column), None)
        if title is None or operations_column is None or variant_column is None:
            continue
        begin = variant_column.find(_BACKTICK)
        end = variant_column.find(_BACKTICK, begin + 1)
        if end < 0:
            raise ValidationError(
                f"nanobench variant has an unmatched backtick: {original_line!r}")
        variant = variant_column[begin + 1:end]
        if not variant:
            raise ValidationError("nanobench variant name must not be empty")
        try:
            operations_per_second = float(
                columns[operations_column].replace(",", ""))
        except (IndexError, ValueError) as error:
            raise ValidationError(
                f"nanobench row has no parseable op/s value: {original_line!r}") \
                from error
        if (not math.isfinite(operations_per_second) or
                operations_per_second <= 0.0):
            raise ValidationError(
                f"nanobench op/s must be finite and positive: {original_line!r}")
        key = (title, variant)
        if key in rows:
            raise ValidationError(f"duplicate nanobench row: {key!r}")
        rows[key] = NanobenchRow(
            title=title,
            variant=variant,
            operations_per_second=operations_per_second,
        )
    if not rows:
        raise ValidationError(
            "no nanobench rows parsed; child output or adapter contract changed")
    return rows


def select_pair(
        rows: Mapping[tuple[str, str], NanobenchRow], candidate: str,
        peer: str, *, title: str | None = None) -> tuple[NanobenchRow, NanobenchRow]:
    """Select exactly one candidate/peer pair, optionally under a fixed title."""
    if not candidate or not peer:
        raise ValidationError("candidate and peer row names must be non-empty")
    titles = []
    for row_title, variant in rows:
        if variant == candidate and (title is None or row_title == title):
            if (row_title, peer) in rows:
                titles.append(row_title)
    unique_titles = sorted(set(titles))
    if not unique_titles:
        scope = "any title" if title is None else repr(title)
        raise ValidationError(
            f"nanobench pair did not match under {scope}: "
            f"{candidate!r} / {peer!r}")
    if len(unique_titles) != 1:
        raise ValidationError(
            f"nanobench pair is ambiguous across titles {unique_titles!r}: "
            f"{candidate!r} / {peer!r}")
    matched_title = unique_titles[0]
    return rows[(matched_title, candidate)], rows[(matched_title, peer)]


def require_consistent_row_sets(
        runs: Iterable[Mapping[tuple[str, str], NanobenchRow]]) -> None:
    """Reject partial process samples instead of silently lowering N."""
    materialized = list(runs)
    if not materialized:
        raise ValidationError("at least one nanobench run is required")
    expected = set(materialized[0])
    if not expected:
        raise ValidationError("first nanobench run contains no rows")
    for index, run in enumerate(materialized[1:], start=2):
        actual = set(run)
        if actual != expected:
            raise ValidationError(
                f"nanobench row set changed in run {index}: "
                f"missing={sorted(expected - actual)!r}, "
                f"extra={sorted(actual - expected)!r}")
