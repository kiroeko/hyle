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

    intrusive_specs, intrusive_files = reporter.module_scope(
        "container_intrusive", files)
    if tuple(intrusive_specs) != reporter.MODULE_BOUNDARIES["container-intrusive"]:
        raise AssertionError("container_intrusive canonical boundary drifted")
    for header in (
        "include/hyle/container/detail/intrusive_member.h",
        "include/hyle/container/intrusive_forward_list.h",
        "include/hyle/container/intrusive_hlist.h",
        "include/hyle/container/intrusive_list.h",
    ):
        if header not in intrusive_files:
            raise AssertionError(
                f"container_intrusive boundary lost implementation header: {header}")

    util_core_specs, util_core_files = reporter.module_scope("util_core", files)
    if tuple(util_core_specs) != reporter.MODULE_BOUNDARIES["util-core"]:
        raise AssertionError("util_core canonical boundary drifted")
    for header in (
        "include/hyle/util/borrow.h",
        "include/hyle/util/function_ref.h",
    ):
        if header not in util_core_files:
            raise AssertionError(f"util_core boundary lost public header: {header}")

    util_runtime_specs, util_runtime_files = reporter.module_scope(
        "util_runtime", files)
    if tuple(util_runtime_specs) != reporter.MODULE_BOUNDARIES["util-runtime"]:
        raise AssertionError("util_runtime canonical boundary drifted")
    for implementation in (
        "include/hyle/util/demangle.h",
        "include/hyle/util/registry.h",
        "src/util_runtime/demangle.cc",
    ):
        if implementation not in util_runtime_files:
            raise AssertionError(
                f"util_runtime boundary lost implementation: {implementation}")

    evolving_io = (
        "include/hyle/io/file.h",
        "include/hyle/io/proactor.h",
        "src/io/file.cc",
    )
    io_specs, io_files = reporter.module_scope("io", evolving_io)
    if io_specs != ["include/hyle/io", "src/io"]:
        raise AssertionError(f"conventional IO boundary drifted: {io_specs!r}")
    if "src/io/file.cc" not in io_files:
        raise AssertionError("a newly added conventional source root escaped IO scope")

    build_specs, build_files = reporter.module_scope(
        "build-package",
        (
            ".github/workflows/build-system.yml",
            "CMakeLists.txt",
            "CMakePresets.json",
            "cmake/HylePackage.cmake",
            "hyle-offline-deps.cmake",
            "scripts/check-docs.py",
            "scripts/header-self-contained.py",
            "scripts/hyle-unix.sh",
            "scripts/hyle-windows.ps1",
            "scripts/validation-select.py",
            "scripts/validation-graph.cmake",
            "scripts/validation-policy.json",
        ),
    )
    if tuple(build_specs) != reporter.MODULE_BOUNDARIES["build-package"]:
        raise AssertionError("build/package canonical boundary drifted")
    if "cmake/HylePackage.cmake" not in build_files:
        raise AssertionError("build/package boundary lost package implementation")
    if "scripts/validation-select.py" not in build_files:
        raise AssertionError("build/package boundary lost validation selector")


if __name__ == "__main__":
    main()
