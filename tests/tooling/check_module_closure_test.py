#!/usr/bin/env python3
"""Focused contract tests for scripts/check-module-closure.py."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


def load_checker():
    script = Path(__file__).parents[2] / "scripts" / "check-module-closure.py"
    spec = importlib.util.spec_from_file_location("hyle_module_closure", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_error(errors: list[str], fragment: str) -> None:
    if not any(fragment in error for error in errors):
        raise AssertionError(f"missing diagnostic {fragment!r}: {errors!r}")


def main() -> None:
    checker = load_checker()
    root = Path(__file__).parents[2]
    registry_path = root / "docs" / "implementation" / "module-status.json"
    status_path = root / "docs" / "implementation" / "status.md"
    data = json.loads(registry_path.read_text(encoding="utf-8"))

    errors = checker.validate_registry(data, root)
    if errors:
        raise AssertionError(f"canonical registry rejected: {errors!r}")
    status_errors = checker.validate_status_page(
        data, status_path.read_text(encoding="utf-8"))
    if status_errors:
        raise AssertionError(f"canonical status projection rejected: {status_errors!r}")

    actual_modules = [record["module"] for record in data["modules"]]
    if actual_modules != checker.declared_components(root):
        raise AssertionError("registry stopped following the central component graph")

    p1_engineering_close = copy.deepcopy(data)
    intrusive = next(
        record for record in p1_engineering_close["modules"]
        if record["module"] == "container_intrusive")
    intrusive["product_state"] = "CLOSED"
    intrusive["engineering_performance"]["verdict"] = "PASS"
    intrusive["gate_verdicts"]["engineering_performance"] = "PASS"
    intrusive["gate_verdicts"]["red_team"] = "PASS"
    errors = checker.validate_registry(p1_engineering_close, root)
    if errors:
        raise AssertionError(
            "P1 engineering PASS was blocked by independent Formal PENDING: "
            f"{errors!r}")

    p1_false_close = copy.deepcopy(data)
    intrusive = next(
        record for record in p1_false_close["modules"]
        if record["module"] == "container_intrusive")
    intrusive["product_state"] = "CLOSED"
    intrusive["end_commit"] = intrusive["candidate_commit"]
    errors = checker.validate_registry(p1_false_close, root)
    require_error(errors, "CLOSED requires engineering verdict PASS")
    require_error(errors, "CLOSED has non-passing gates")

    missing_loc = copy.deepcopy(data)
    core = next(
        record for record in missing_loc["modules"]
        if record["module"] == "core")
    del core["loc"]["module_implementation"]
    errors = checker.validate_registry(missing_loc, root)
    require_error(errors, "missing LOC fields")

    duplicate = copy.deepcopy(data)
    duplicate["modules"].append(copy.deepcopy(duplicate["modules"][0]))
    errors = checker.validate_registry(duplicate, root)
    require_error(errors, "duplicate module ids")
    require_error(errors, "differs from HyleComponents.cmake")

    absolute_evidence = copy.deepcopy(data)
    core = next(
        record for record in absolute_evidence["modules"]
        if record["module"] == "core")
    core["closure_evidence"] = "C:/private/core.log"
    errors = checker.validate_registry(absolute_evidence, root)
    require_error(errors, "must not be an absolute path")

    illegal = copy.deepcopy(data)
    core = next(
        record for record in illegal["modules"]
        if record["module"] == "core")
    core["product_state"] = "QUEUED"
    errors = checker.validate_transitions(data, illegal)
    require_error(errors, "illegal transition 'CLOSED' -> 'QUEUED'")

    drifted_status = status_path.read_text(encoding="utf-8").replace(
        "| core | CLOSED |", "| core | QUEUED |", 1)
    errors = checker.validate_status_page(data, drifted_status)
    require_error(errors, "differs from module-status.json")


if __name__ == "__main__":
    main()
