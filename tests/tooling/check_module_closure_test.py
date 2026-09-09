#!/usr/bin/env python3
"""Focused regression tests for the executable module-closure contract."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile


def load_checker():
    script = Path(__file__).parents[2] / "scripts" / "check-module-closure.py"
    spec = importlib.util.spec_from_file_location("hyle_check_module_closure", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_error(errors: list[str], fragment: str) -> None:
    if not any(fragment in error for error in errors):
        raise AssertionError(f"missing error containing {fragment!r}: {errors!r}")


def record(data, component: str):
    return next(item for item in data["modules"] if item["module"] == component)


def transition(before_state: str, after_state: str):
    return (
        {"modules": [{"module": "sample", "state": before_state}]},
        {"modules": [{"module": "sample", "state": after_state}]},
    )


def main() -> None:
    checker = load_checker()
    root = Path(__file__).parents[2]
    registry_path = root / "docs" / "implementation" / "module-status.json"
    status_path = root / "docs" / "implementation" / "status.md"
    canonical = checker.load_registry(registry_path)

    errors = checker.validate_registry(canonical, root)
    if errors:
        raise AssertionError(f"canonical registry rejected: {errors!r}")
    states = [item["state"] for item in canonical["modules"]]
    if states.count("CLOSED") != 26 or states.count("QUEUED") != 2:
        raise AssertionError(f"unexpected canonical state distribution: {states!r}")
    serialized = repr(canonical)
    for retired in ("VERIFIED", "FROZEN", "freeze_confirmation", "formal_performance"):
        if retired in serialized:
            raise AssertionError(f"retired governance term remains in registry: {retired}")

    for item in canonical["modules"]:
        if item["state"] != "CLOSED":
            continue
        if set(item["candidate"]) != {"commit", "tree"}:
            raise AssertionError(f"candidate identity drifted for {item['module']}")
        if item["performance"]["verdict"] != "PASS":
            raise AssertionError(f"performance verdict drifted for {item['module']}")
        if set(item["gates"]) != checker.GATE_NAMES:
            raise AssertionError(f"gate set drifted for {item['module']}")

    malformed = copy.deepcopy(canonical)
    malformed["unexpected"] = True
    require_error(checker.validate_registry(malformed, root), "unknown root fields")

    malformed = copy.deepcopy(canonical)
    malformed["updated_at"] = "2026-02-30"
    require_error(checker.validate_registry(malformed, root), "valid calendar date")

    malformed = copy.deepcopy(canonical)
    core = record(malformed, "core")
    core["product_state"] = core["state"]
    require_error(checker.validate_registry(malformed, root), "retired registry fields")

    malformed = copy.deepcopy(canonical)
    record(malformed, "core")["state"] = "VERIFIED"
    require_error(checker.validate_registry(malformed, root), "invalid state")

    malformed = copy.deepcopy(canonical)
    record(malformed, "core")["candidate"]["tree"] = "1234"
    require_error(checker.validate_registry(malformed, root), "candidate.tree")

    malformed = copy.deepcopy(canonical)
    record(malformed, "core")["record"] = "docs/implementation"
    require_error(checker.validate_registry(malformed, root), "not a repository file")

    malformed = copy.deepcopy(canonical)
    record(malformed, "core")["performance"]["material_margin_pct"] = 5
    require_error(checker.validate_registry(malformed, root), "P0 must not")

    malformed = copy.deepcopy(canonical)
    record(malformed, "error")["performance"].pop("material_margin_pct")
    require_error(checker.validate_registry(malformed, root), "positive material_margin_pct")

    malformed = copy.deepcopy(canonical)
    record(malformed, "error")["performance"]["verdict"] = "INCONCLUSIVE"
    require_error(checker.validate_registry(malformed, root), "verdict PASS")

    malformed = copy.deepcopy(canonical)
    record(malformed, "error")["gates"]["correctness"] = "FAIL"
    require_error(checker.validate_registry(malformed, root), "non-passing gates")

    malformed = copy.deepcopy(canonical)
    malformed["modules"].append(copy.deepcopy(malformed["modules"][0]))
    require_error(checker.validate_registry(malformed, root), "duplicate module ids")

    before, after = transition("CANDIDATE", "CLOSED")
    if errors := checker.validate_transitions(before, after):
        raise AssertionError(f"legal CANDIDATE -> CLOSED rejected: {errors!r}")
    before, after = transition("CLOSED", "QUEUED")
    require_error(checker.validate_transitions(before, after), "illegal transition")
    before, after = transition("QUEUED", "BLOCKED")
    require_error(checker.validate_transitions(before, after), "illegal transition")

    before = {"modules": [{"module": "a", "state": "QUEUED"}]}
    after = {
        "modules": [
            {"module": "a", "state": "QUEUED"},
            {"module": "b", "state": "SCOPED"},
        ]
    }
    require_error(checker.validate_transitions(before, after), "must enter as 'QUEUED'")
    require_error(
        checker.validate_transitions(
            {"modules": [{"module": "a", "state": "QUEUED"}]},
            {"modules": []},
        ),
        "component removal",
    )
    require_error(
        checker.validate_transitions(
            {
                "modules": [
                    {"module": "a", "state": "QUEUED"},
                    {"module": "b", "state": "QUEUED"},
                ]
            },
            {
                "modules": [
                    {"module": "b", "state": "QUEUED"},
                    {"module": "a", "state": "QUEUED"},
                ]
            },
        ),
        "reordering",
    )

    status_text = status_path.read_text(encoding="utf-8")
    if errors := checker.validate_status_page(canonical, status_text):
        raise AssertionError(f"canonical status projection rejected: {errors!r}")
    drifted = status_text.replace("| core | CLOSED |", "| core | QUEUED |", 1)
    require_error(checker.validate_status_page(canonical, drifted), "differs")
    with tempfile.TemporaryDirectory(prefix="hyle-status-test-") as temp:
        copy_path = Path(temp) / "status.md"
        copy_path.write_text(drifted, encoding="utf-8")
        checker.write_status_page(canonical, copy_path)
        repaired = copy_path.read_text(encoding="utf-8")
        if errors := checker.validate_status_page(canonical, repaired):
            raise AssertionError(f"status writer failed to repair drift: {errors!r}")

    historical = checker.load_registry_ref(root, "HEAD")
    if not isinstance(historical.get("modules"), list):
        raise AssertionError("Git registry loader lost modules")


if __name__ == "__main__":
    main()
