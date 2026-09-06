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

    # Contract tests must remain valid while any particular module is moving
    # through a legal closure round. Pick a terminal receipt for mutations that
    # specifically exercise CLOSED-only fields instead of assuming Core is
    # permanently CLOSED.
    terminal_record = next(
        record for record in data["modules"]
        if record["product_state"] == "CLOSED"
    )
    terminal_module = terminal_record["module"]
    p1_terminal_record = next(
        record for record in data["modules"]
        if record["product_state"] == "CLOSED" and
        record.get("performance_tier") == "P1"
    )
    p1_terminal_module = p1_terminal_record["module"]

    p1_engineering_close = copy.deepcopy(data)
    p1_record = next(
        record for record in p1_engineering_close["modules"]
        if record["module"] == p1_terminal_module)
    p1_record["engineering_performance"]["verdict"] = "PASS"
    p1_record["gate_verdicts"]["engineering_performance"] = "PASS"
    p1_record["gate_verdicts"]["red_team"] = "PASS"
    errors = checker.validate_registry(p1_engineering_close, root)
    if errors:
        raise AssertionError(
            "P1 engineering PASS was blocked by independent Formal PENDING: "
            f"{errors!r}")

    p1_false_close = copy.deepcopy(data)
    p1_record = next(
        record for record in p1_false_close["modules"]
        if record["module"] == p1_terminal_module)
    p1_record["engineering_performance"]["verdict"] = "PLANNED"
    p1_record["gate_verdicts"]["engineering_performance"] = "INCONCLUSIVE"
    p1_record["gate_verdicts"]["red_team"] = "PASS"
    errors = checker.validate_registry(p1_false_close, root)
    require_error(errors, "CLOSED requires engineering verdict PASS")
    require_error(errors, "CLOSED has non-passing gates")

    missing_loc = copy.deepcopy(data)
    core = next(
        record for record in missing_loc["modules"]
        if record["module"] == terminal_module)
    del core["loc"]["module_implementation"]
    errors = checker.validate_registry(missing_loc, root)
    require_error(errors, "missing LOC fields")

    missing_anchor = copy.deepcopy(data)
    core = next(
        record for record in missing_anchor["modules"]
        if record["module"] == terminal_module)
    del core["registry_anchor_commit"]
    errors = checker.validate_registry(missing_anchor, root)
    require_error(errors, "registry_anchor_commit")

    frozen_record = next(
        record for record in data["modules"]
        if isinstance(record.get("freeze_confirmation"), dict)
    )

    invalid_freeze_marker = copy.deepcopy(data)
    record = next(
        item for item in invalid_freeze_marker["modules"]
        if item["module"] == frozen_record["module"])
    record["freeze_confirmation"]["marker"] = "DONE"
    errors = checker.validate_registry(invalid_freeze_marker, root)
    require_error(errors, "freeze_confirmation.marker must be 'FROZEN'")

    invalid_freeze_identity = copy.deepcopy(data)
    record = next(
        item for item in invalid_freeze_identity["modules"]
        if item["module"] == frozen_record["module"])
    record["freeze_confirmation"]["reviewed_tree"] = "1234"
    errors = checker.validate_registry(invalid_freeze_identity, root)
    require_error(errors, "full 40-digit lowercase Git id")

    frozen_active = copy.deepcopy(data)
    record = next(
        item for item in frozen_active["modules"]
        if item["module"] == frozen_record["module"])
    record["product_state"] = "VERIFIED"
    errors = checker.validate_registry(frozen_active, root)
    require_error(errors, "freeze_confirmation is only valid for CLOSED")

    short_identity = copy.deepcopy(data)
    core = next(
        record for record in short_identity["modules"]
        if record["module"] == terminal_module)
    core["baseline_commit"] = core["baseline_commit"][:7]
    errors = checker.validate_registry(short_identity, root)
    require_error(errors, "full 40-digit lowercase Git id")

    ghost_machine = copy.deepcopy(data)
    intrusive = next(
        record for record in ghost_machine["modules"]
        if record["module"] == "container_intrusive")
    intrusive["engineering_performance"]["machine_profile"] = "GHOST"
    errors = checker.validate_registry(ghost_machine, root)
    require_error(errors, "unknown engineering machine profile")

    wrong_p2_instrument = copy.deepcopy(data)
    intrusive = next(
        record for record in wrong_p2_instrument["modules"]
        if record["module"] == "container_intrusive")
    intrusive["performance_tier"] = "P2"
    intrusive["engineering_performance"]["mode"] = "P2_CONSUMER"
    intrusive["engineering_performance"]["instrument_profile"] = (
        "closure-correctness-v1")
    errors = checker.validate_registry(wrong_p2_instrument, root)
    require_error(errors, "P2 must use closure-perf-v1")

    verified_planned = copy.deepcopy(data)
    p1_record = next(
        record for record in verified_planned["modules"]
        if record["module"] == p1_terminal_module)
    p1_record["product_state"] = "VERIFIED"
    p1_record["engineering_performance"]["verdict"] = "PLANNED"
    errors = checker.validate_registry(verified_planned, root)
    require_error(errors, "cannot retain a PLANNED engineering verdict")

    closed_red_team_na = copy.deepcopy(p1_engineering_close)
    p1_record = next(
        record for record in closed_red_team_na["modules"]
        if record["module"] == p1_terminal_module)
    p1_record["gate_verdicts"]["red_team"] = "NOT_APPLICABLE"
    errors = checker.validate_registry(closed_red_team_na, root)
    require_error(errors, "CLOSED has non-passing gates: red_team")

    task_is_directory = copy.deepcopy(data)
    core = next(
        record for record in task_is_directory["modules"]
        if record["module"] == terminal_module)
    core["closure_task"] = "docs/implementation/tasks/core"
    errors = checker.validate_registry(task_is_directory, root)
    require_error(errors, "must name a repository file")

    invalid_date = copy.deepcopy(data)
    invalid_date["updated_at"] = "2026-99-99"
    errors = checker.validate_registry(invalid_date, root)
    require_error(errors, "valid calendar date")

    duplicate = copy.deepcopy(data)
    duplicate["modules"].append(copy.deepcopy(duplicate["modules"][0]))
    errors = checker.validate_registry(duplicate, root)
    require_error(errors, "duplicate module ids")
    require_error(errors, "differs from HyleComponents.cmake")

    absolute_evidence = copy.deepcopy(data)
    core = next(
        record for record in absolute_evidence["modules"]
        if record["module"] == terminal_module)
    core["closure_evidence"] = "C:/private/core.log"
    errors = checker.validate_registry(absolute_evidence, root)
    require_error(errors, "must not be an absolute path")

    illegal = copy.deepcopy(data)
    core = next(
        record for record in illegal["modules"]
        if record["module"] == terminal_module)
    core["product_state"] = "QUEUED"
    errors = checker.validate_transitions(data, illegal)
    require_error(errors, "illegal transition 'CLOSED' -> 'QUEUED'")

    queued_before = copy.deepcopy(data)
    error = next(
        record for record in queued_before["modules"]
        if record["module"] == "error")
    error["product_state"] = "QUEUED"
    queued_to_blocked = copy.deepcopy(queued_before)
    error = next(
        record for record in queued_to_blocked["modules"]
        if record["module"] == "error")
    error["product_state"] = "BLOCKED"
    errors = checker.validate_transitions(queued_before, queued_to_blocked)
    require_error(errors, "illegal transition 'QUEUED' -> 'BLOCKED'")

    queued_addition = copy.deepcopy(data)
    queued_addition["modules"].insert(
        1,
        {
            "module": "new_factor",
            "product_state": "QUEUED",
            "summary": "Registered before its closure round starts.",
        },
    )
    errors = checker.validate_transitions(data, queued_addition)
    if errors:
        raise AssertionError(f"legal QUEUED component addition rejected: {errors!r}")

    scoped_addition = copy.deepcopy(queued_addition)
    scoped_addition["modules"][1]["product_state"] = "SCOPED"
    errors = checker.validate_transitions(data, scoped_addition)
    require_error(errors, "new component 'new_factor' must enter as 'QUEUED'")

    removed_component = copy.deepcopy(data)
    removed_component["modules"].pop()
    errors = checker.validate_transitions(data, removed_component)
    require_error(errors, "does not allow component removal or rename")

    reordered_components = copy.deepcopy(data)
    reordered_components["modules"][0:2] = reversed(
        reordered_components["modules"][0:2])
    errors = checker.validate_transitions(data, reordered_components)
    require_error(errors, "does not allow reordering existing components")

    drifted_status = status_path.read_text(encoding="utf-8").replace(
        f"| {terminal_module} | CLOSED |",
        f"| {terminal_module} | QUEUED |",
        1,
    )
    errors = checker.validate_status_page(data, drifted_status)
    require_error(errors, "differs from module-status.json")


if __name__ == "__main__":
    main()
