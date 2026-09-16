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
        "include/hyle/io/buffer.h",
        "include/hyle/io/capabilities.h",
        "include/hyle/io/detail/io_context_access.h",
        "include/hyle/io/detail/net.h",
        "include/hyle/io/detail/net_budget.h",
        "include/hyle/io/detail/net_cancel.h",
        "include/hyle/io/detail/net_multishot.h",
        "include/hyle/io/detail/net_timeout.h",
        "include/hyle/io/detail/proactor.h",
        "include/hyle/io/dns.h",
        "include/hyle/io/endpoint.h",
        "include/hyle/io/file.h",
        "include/hyle/io/http.h",
        "include/hyle/io/http2.h",
        "include/hyle/io/io_context.h",
        "include/hyle/io/io_uring.h",
        "include/hyle/io/local.h",
        "include/hyle/io/operation.h",
        "include/hyle/io/signal.h",
        "include/hyle/io/tcp.h",
        "include/hyle/io/tls.h",
        "include/hyle/io/transfer.h",
        "include/hyle/io/udp.h",
        "src/io_core/file.cc",
        "src/io_tls/tls.cc",
        "src/io_http/http.cc",
        "src/io_http2/http2.cc",
        "src/io_transfer/transfer.cc",
    )
    io_specs, io_files = reporter.module_scope("io_core", evolving_io)
    if "src/io_core" not in io_specs or "include/hyle/io/file.h" not in io_specs:
        raise AssertionError(f"split IO Core boundary drifted: {io_specs!r}")
    if "src/io_core/file.cc" not in io_files:
        raise AssertionError("a newly added IO Core source escaped its factor scope")
    for core_header in (
        "include/hyle/io/buffer.h",
        "include/hyle/io/endpoint.h",
        "include/hyle/io/local.h",
        "include/hyle/io/operation.h",
    ):
        if core_header not in io_files:
            raise AssertionError(f"IO Core boundary lost public header: {core_header}")
    for protocol_header in (
        "include/hyle/io/http.h",
        "include/hyle/io/http2.h",
        "include/hyle/io/tls.h",
        "include/hyle/io/transfer.h",
    ):
        if protocol_header in io_files:
            raise AssertionError(
                f"IO Core boundary absorbed protocol factor: {protocol_header}")

    for factor, header, source in (
        ("io_tls", "include/hyle/io/tls.h", "src/io_tls/tls.cc"),
        ("io_http", "include/hyle/io/http.h", "src/io_http/http.cc"),
        ("io_http2", "include/hyle/io/http2.h", "src/io_http2/http2.cc"),
        ("io_transfer", "include/hyle/io/transfer.h", "src/io_transfer/transfer.cc"),
    ):
        factor_specs, factor_files = reporter.module_scope(factor, evolving_io)
        expected_specs = reporter.MODULE_BOUNDARIES[factor.replace("_", "-")]
        if tuple(factor_specs) != expected_specs:
            raise AssertionError(f"{factor} canonical boundary drifted")
        if header not in factor_files or source not in factor_files:
            raise AssertionError(f"{factor} boundary lost its public header or source")

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

    registry = reporter.snapshot_registry(root, None)
    graph = reporter.snapshot_component_graph(root, "WORKTREE")
    closed_modules, closed_files = reporter.registry_state_scope(
        "CLOSED", files, registry, graph
    )
    expected_closed = [
        record["module"]
        for record in registry["modules"]
        if reporter.registry_record_state(record) == "CLOSED"
    ]
    if closed_modules != expected_closed:
        raise AssertionError("CLOSED LOC selection lost registry order or membership")
    # State selection must not freeze an old project's construction queue.
    # Use an explicit mixed-state fixture, including a real native .mm owner.
    native = "src/compute_route_apple/apple_environment.mm"
    fixture_files = (
        "src/compute/completion.cc", "src/graph/graph.cc", native,
        "tests/compute_route_apple/example.mm",
    )
    fixture_registry = {"modules": [
        {"module": "compute", "state": "CLOSED"},
        {"module": "graph", "state": "QUEUED"},
        {"module": "compute_route_apple", "state": "AUDITED"},
    ]}
    for state, module, expected in (
        ("CLOSED", "compute", "src/compute/completion.cc"),
        ("QUEUED", "graph", "src/graph/graph.cc"),
        ("AUDITED", "compute_route_apple", native),
    ):
        owners, chosen = reporter.registry_state_scope(
            state, fixture_files, fixture_registry, graph)
        if owners != [module] or chosen != [expected]:
            raise AssertionError(f"native owner/state selection mismatch: {owners}, {chosen}")
    cloc = reporter.find_cloc()
    actual = reporter.cloc_measure(cloc, root, [native], None)
    selected = reporter.cloc_measure(cloc, root, [native], reporter.NATIVE_LANGUAGES)
    if actual[0] == 0 or actual[1] != 1 or selected != actual:
        raise AssertionError(f"Objective-C++ implementation is silently omitted: {selected} != {actual}")


if __name__ == "__main__":
    main()
