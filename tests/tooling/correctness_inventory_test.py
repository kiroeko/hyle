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
    report = tool.build_inventory(root, "WORKTREE")

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

    expected_journeys = {
        "tests/correctness/async_runtime_journey_test.cc",
        "tests/correctness/data_preparation_journey_test.cc",
        "tests/correctness/model_file_lifecycle_journey_test.cc",
        "tests/correctness/service_connection_journey_test.cc",
        "tests/correctness/working_process_journey_test.cc",
    }
    journeys = {row["path"] for row in report["project_tests"]}
    if journeys != expected_journeys:
        raise AssertionError(f"project correctness journeys drifted: {journeys!r}")
    if not all(row["registered"] for row in report["project_tests"]):
        raise AssertionError("a project correctness journey is not registered")
    if report["summary"]["project_test_sources"] != 5:
        raise AssertionError("expected the five project correctness journeys")

    products = {row["product"]: row for row in report["products"]}
    if not products["io"]["all_factors_closed"]:
        raise AssertionError("the current IO product closure should be fully CLOSED")
    if products["compute"]["all_factors_closed"]:
        raise AssertionError("QUEUED Compute must not be reported as qualified")
    if products["graph"]["all_factors_closed"]:
        raise AssertionError("QUEUED Graph must not be reported as qualified")

    closed_products = tool.closed_product_names(report)
    if "compute" in closed_products or "graph" in closed_products:
        raise AssertionError("a QUEUED product leaked into the CLOSED product projection")
    projected_factors = {
        factor
        for row in report["products"]
        if row["all_factors_closed"]
        for factor in row["factor_closure"]
    }
    if projected_factors != set(factors):
        raise AssertionError(
            "CLOSED product projection does not cover the exact CLOSED factor set: "
            f"projected={sorted(projected_factors)!r}, factors={sorted(factors)!r}"
        )

    for relative in (
        "cmake/HyleCiProductShapeSmoke.cmake",
        "cmake/HyleCiPackageSmoke.cmake",
        "cmake/HyleCiSourcePackageSmoke.cmake",
    ):
        smoke = (root / relative).read_text(encoding="utf-8")
        if "HyleCiClosedProducts.cmake" not in smoke:
            raise AssertionError(f"{relative} bypasses the CLOSED product projection")
        if "HYLE_COMPONENTS=all" in smoke:
            raise AssertionError(f"{relative} still admits QUEUED products")

    windows_driver = (root / "scripts/hyle-windows.ps1").read_text(encoding="utf-8")
    source_action = windows_driver.split("'SourcePackageSmoke' {", 1)[1]
    if "Get-HyleClosedProducts" not in source_action:
        raise AssertionError("Windows source-package driver bypasses CLOSED projection")
    if "HYLE_COMPONENTS=all" in source_action:
        raise AssertionError("Windows source-package driver still admits QUEUED products")


if __name__ == "__main__":
    main()
