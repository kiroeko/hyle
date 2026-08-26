#!/usr/bin/env python3
"""Focused tests for immutable LOC snapshot selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import re
import sys


def load_reporter():
    script = Path(__file__).parents[2] / "scripts" / "loc-report.py"
    spec = importlib.util.spec_from_file_location("hyle_loc_report", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    reporter = load_reporter()
    root = Path(__file__).parents[2]
    commit, files = reporter.tracked_commit_files(root, "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise AssertionError(f"HEAD did not resolve to a full commit: {commit!r}")
    if "include/hyle/core/cache.h" not in files:
        raise AssertionError("immutable Git file list lost Core headers")

    _, selected = reporter.module_scope("core", files)
    if not selected:
        raise AssertionError("Core immutable module selection is empty")

    synthetic = (
        "include/hyle/core/cache.h",
        "src/core/CMakeLists.txt",
        "tests/core/cache_test.cc",
        "third_party/vendor/src/not_hyle.cc",
        "third_party/vendor/tests/not_hyle_test.cc",
    )
    self_files = reporter.files_under(synthetic, reporter.SELF_WITH_TESTS_ROOTS)
    if self_files != list(synthetic[:3]):
        raise AssertionError(
            "self-owned roots admitted a nested vendor src/tests directory: "
            f"{self_files!r}"
        )

    evolving_io = (
        "include/hyle/io/file.h",
        "include/hyle/c/io.h",
        "include/hyle/c/io_compat.h",
        "src/io/file.cc",
    )
    io_specs, io_files = reporter.module_scope("io", evolving_io)
    if "include/hyle/c/io_compat.h" not in io_specs:
        raise AssertionError("a newly added conventional C header escaped IO scope")
    if "src/io/file.cc" not in io_files:
        raise AssertionError("a newly added conventional source root escaped IO scope")


if __name__ == "__main__":
    main()
