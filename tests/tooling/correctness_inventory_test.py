#!/usr/bin/env python3
"""Focused contract tests for the CLOSED-factor correctness inventory."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_inventory():
    script = Path(__file__).parents[2] / "scripts" / "correctness-inventory.py"
    spec = importlib.util.spec_from_file_location("hyle_correctness_inventory", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tool = load_inventory()
    root = Path(__file__).parents[2]
    report = tool.build_inventory(root, "HEAD")

    if report["schema"] != "hyle.correctness-inventory/v1":
        raise AssertionError("correctness inventory schema drifted")
    if report["summary"]["closed_factors"] != 29:
        raise AssertionError("expected the current 29 CLOSED factors")
    if report["issues"]:
        raise AssertionError(f"baseline correctness inventory has issues: {report['issues']!r}")

    factors = {row["factor"]: row for row in report["factors"]}
    if "include/hyle/mem/allocator.h" not in factors["mem"]["public_headers"]:
        raise AssertionError("Mem public-header ownership was not projected")
    if factors["io_websocket"]["totals"]["test_declarations"] == 0:
        raise AssertionError("WebSocket tests were not discovered")
    if report["cross_cutting_public_headers"] != ["include/hyle/obs/trace.h"]:
        raise AssertionError("cross-cutting trace-header exception drifted")

    products = {row["product"]: row for row in report["products"]}
    if not products["io"]["all_factors_closed"]:
        raise AssertionError("the current IO product closure should be fully CLOSED")
    if products["compute"]["all_factors_closed"]:
        raise AssertionError("QUEUED Compute must not be reported as qualified")
    if products["graph"]["all_factors_closed"]:
        raise AssertionError("QUEUED Graph must not be reported as qualified")


if __name__ == "__main__":
    main()
