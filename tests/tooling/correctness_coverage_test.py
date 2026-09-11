#!/usr/bin/env python3
"""Focused pure-data tests for the correctness coverage discovery tool."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def load_tool():
    script = Path(__file__).parents[2] / "scripts" / "correctness-coverage.py"
    spec = importlib.util.spec_from_file_location("hyle_correctness_coverage", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    tool = load_tool()
    root = Path("/repo")
    payloads = [{
        "current_working_directory": "/build",
        "files": [{
            "file": "/repo/include/hyle/core/cache.h",
            "lines": [
                {"line_number": 4, "count": 2, "branches": [
                    {"count": 2, "source_block_id": 1,
                     "destination_block_id": 2, "fallthrough": True,
                     "throw": False},
                    {"count": 0, "source_block_id": 1,
                     "destination_block_id": 3, "fallthrough": False,
                     "throw": False},
                ]},
                {"line_number": 5, "count": 0, "branches": []},
                {"line_number": 7, "count": 0, "branches": []},
            ],
            "functions": [
                {"name": "hit", "start_line": 4, "execution_count": 2},
                {"name": "miss", "start_line": 7, "execution_count": 0},
            ],
        }, {
            "file": "/usr/include/vector",
            "lines": [{"line_number": 1, "count": 3, "branches": []}],
            "functions": [],
        }],
    }]
    report = tool.aggregate_gcov(
        root,
        payloads,
        ["core"],
        {
            "include/hyle/core/cache.h": "core",
            "include/hyle/core/compiler.h": "core",
        },
    )
    summary = report["summary"]
    if (summary["executable_lines"], summary["covered_lines"]) != (3, 1):
        raise AssertionError("line aggregation drifted")
    if (summary["branches"], summary["covered_branches"]) != (2, 1):
        raise AssertionError("branch aggregation drifted")
    if report["files"][0]["zero_hit_line_ranges"] != ["5", "7"]:
        raise AssertionError("zero-hit line ranges drifted")
    if report["files"][0]["zero_hit_functions"] != ["miss"]:
        raise AssertionError("zero-hit function discovery drifted")
    if report["unobserved_product_files"] != ["include/hyle/core/compiler.h"]:
        raise AssertionError("unobserved file discovery drifted")
    if tool._line_ranges([8, 7, 3, 2, 1, 8]) != ["1-3", "7-8"]:
        raise AssertionError("line range compaction drifted")


if __name__ == "__main__":
    main()
