#!/usr/bin/env python3
"""Validate HYLE's canonical module-closure registry and status projection."""

from __future__ import annotations

import argparse
import datetime
import json
import posixpath
import re
import sys
from pathlib import Path
from typing import Any


PRODUCT_STATES = {
    "QUEUED",
    "SCOPED",
    "AUDITED",
    "CANDIDATE",
    "VERIFIED",
    "CLOSED",
    "BLOCKED",
    "REOPENED",
}
PERFORMANCE_TIERS = {"P0", "P1", "P2"}
ENGINEERING_VERDICTS = {
    "PLANNED",
    "PASS",
    "FAIL",
    "INCONCLUSIVE",
}
ENGINEERING_MODES = {"STRUCTURAL", "P1_ENGINEERING", "P2_CONSUMER"}
FORMAL_STATES = {"NONE", "PENDING", "CERTIFIED", "FAILED", "EXPIRED"}
GATE_NAMES = {
    "scope",
    "architecture",
    "correctness",
    "delivery",
    "usability_docs",
    "red_team",
    "engineering_performance",
}
GATE_VERDICTS = {"PASS", "FAIL", "INCONCLUSIVE"}
LOC_FIELDS = {
    "repository_code",
    "self_with_tests",
    "self_without_tests",
    "module_implementation",
}
STATUS_TABLE_BEGIN = "<!-- BEGIN GENERATED MODULE STATUS -->"
STATUS_TABLE_END = "<!-- END GENERATED MODULE STATUS -->"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
COMPONENT_RE = re.compile(r"_hyle_declare_component\(\s*([a-z][a-z0-9_]*)")

ALLOWED_TRANSITIONS = {
    "QUEUED": {"QUEUED", "SCOPED"},
    "SCOPED": {"SCOPED", "AUDITED", "BLOCKED"},
    "AUDITED": {"AUDITED", "CANDIDATE", "BLOCKED"},
    "CANDIDATE": {"CANDIDATE", "AUDITED", "VERIFIED", "BLOCKED"},
    "VERIFIED": {"VERIFIED", "CANDIDATE", "CLOSED", "BLOCKED"},
    "CLOSED": {"CLOSED", "REOPENED"},
    "BLOCKED": {"BLOCKED", "SCOPED"},
    "REOPENED": {"REOPENED", "SCOPED"},
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("registry root must be a JSON object")
    return value


def declared_components(root: Path) -> list[str]:
    manifest = root / "cmake" / "HyleComponents.cmake"
    text = manifest.read_text(encoding="utf-8")
    result: list[str] = []
    for component in COMPONENT_RE.findall(text):
        if component == "name" or component in result:
            continue
        result.append(component)
    if not result:
        raise ValueError(f"no components found in {manifest}")
    return result


def declared_performance_profiles(root: Path) -> tuple[set[str], set[str]]:
    registry = root / "scripts" / "perf-machines.json"
    value = json.loads(registry.read_text(encoding="utf-8"))
    instruments = value.get("instrument_profiles")
    machines = value.get("machine_profiles")
    if (not isinstance(instruments, list) or
            any(not isinstance(item, str) for item in instruments)):
        raise ValueError(f"invalid instrument_profiles in {registry}")
    if not isinstance(machines, dict):
        raise ValueError(f"invalid machine_profiles in {registry}")
    docs = value.get("instrument_profile_docs")
    if not isinstance(docs, dict) or set(docs) != set(instruments):
        raise ValueError(
            f"instrument_profile_docs in {registry} must map every instrument exactly")
    for profile, path in docs.items():
        if not isinstance(path, str) or not path or Path(path).is_absolute():
            raise ValueError(
                f"invalid documentation path for instrument {profile!r} in {registry}")
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(
                f"instrument documentation for {profile!r} escapes the repository") from error
        if not resolved.is_file():
            raise ValueError(
                f"instrument documentation for {profile!r} is not a file: {path}")
    return set(instruments), set(machines)


def _require_string(
        record: dict[str, Any], field: str, prefix: str,
        errors: list[str]) -> str | None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}: '{field}' must be a non-empty string")
        return None
    return value


def _check_commit(
        record: dict[str, Any], field: str, prefix: str,
        errors: list[str]) -> None:
    value = _require_string(record, field, prefix, errors)
    if value is not None and COMMIT_RE.fullmatch(value) is None:
        errors.append(f"{prefix}: '{field}' is not a full 40-digit lowercase Git id")


def _check_repo_path(
        root: Path, value: Any, field: str, prefix: str,
        errors: list[str]) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{prefix}: '{field}' must be a non-empty repository path")
        return
    candidate = Path(value)
    if candidate.is_absolute() or re.match(r"^[A-Za-z]:", value):
        errors.append(f"{prefix}: '{field}' must not be an absolute path: {value}")
        return
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{prefix}: '{field}' escapes the repository: {value}")
        return
    if not resolved.exists():
        errors.append(f"{prefix}: '{field}' does not exist: {value}")


def _check_repo_file(
        root: Path, value: Any, field: str, prefix: str,
        errors: list[str]) -> None:
    before = len(errors)
    _check_repo_path(root, value, field, prefix, errors)
    if len(errors) != before or not isinstance(value, str):
        return
    if not (root / value).resolve().is_file():
        errors.append(f"{prefix}: '{field}' must name a repository file: {value}")


def _contains_tbd(value: Any) -> bool:
    if isinstance(value, str):
        return "TBD" in value.upper()
    if isinstance(value, list):
        return any(_contains_tbd(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_tbd(item) for item in value.values())
    return False


def _check_nonempty_string_list(
        record: dict[str, Any], field: str, prefix: str,
        errors: list[str]) -> None:
    value = record.get(field)
    if (not isinstance(value, list) or not value or
            any(not isinstance(item, str) or not item.strip() for item in value)):
        errors.append(f"{prefix}: '{field}' must be a non-empty string array")


def validate_module(
        record: Any, root: Path, machine_profiles: set[str],
        instrument_profiles: set[str], errors: list[str]) -> None:
    if not isinstance(record, dict):
        errors.append("module entry must be a JSON object")
        return
    component = record.get("module")
    prefix = f"module {component!r}"
    if not isinstance(component, str) or not component:
        errors.append("module entry has no non-empty 'module' id")
        return

    state = record.get("product_state")
    if state not in PRODUCT_STATES:
        errors.append(f"{prefix}: invalid product_state {state!r}")
        return

    if state == "QUEUED":
        _require_string(record, "summary", prefix, errors)
        return

    effective_state = state
    if state == "BLOCKED":
        _require_string(record, "blocker", prefix, errors)
        _require_string(record, "blocker_owner", prefix, errors)
        _require_string(record, "resume_condition", prefix, errors)
        _require_string(record, "summary", prefix, errors)
        blocked_from = record.get("blocked_from")
        if blocked_from not in {"SCOPED", "AUDITED", "CANDIDATE", "VERIFIED"}:
            errors.append(
                f"{prefix}: BLOCKED requires blocked_from at an active phase")
        else:
            effective_state = blocked_from

    if state == "REOPENED":
        _require_string(record, "reopen_trigger_id", prefix, errors)
        _require_string(record, "reopen_evidence", prefix, errors)

    for field in ("round_id", "closure_task", "scope", "non_goals"):
        _require_string(record, field, prefix, errors)
    _check_repo_file(root, record.get("closure_task"), "closure_task", prefix, errors)
    _check_commit(record, "baseline_commit", prefix, errors)

    tier = record.get("performance_tier")
    if tier not in PERFORMANCE_TIERS:
        errors.append(f"{prefix}: invalid performance_tier {tier!r}")
    profiles = record.get("validation_machine_profiles")
    if (not isinstance(profiles, list) or not profiles or
            any(not isinstance(item, str) or not item for item in profiles)):
        errors.append(
            f"{prefix}: validation_machine_profiles must be a non-empty string array")
    elif unknown := sorted(set(profiles) - machine_profiles):
        errors.append(f"{prefix}: unknown machine profiles: {', '.join(unknown)}")

    engineering = record.get("engineering_performance")
    if not isinstance(engineering, dict):
        errors.append(f"{prefix}: engineering_performance must be an object")
    else:
        verdict = engineering.get("verdict")
        mode = engineering.get("mode")
        if verdict not in ENGINEERING_VERDICTS:
            errors.append(f"{prefix}: invalid engineering verdict {verdict!r}")
        if mode not in ENGINEERING_MODES:
            errors.append(f"{prefix}: invalid engineering mode {mode!r}")
        instrument = _require_string(
            engineering, "instrument_profile", prefix, errors)
        if instrument is not None and instrument not in instrument_profiles:
            errors.append(f"{prefix}: unknown instrument profile {instrument!r}")
        if effective_state in {"VERIFIED", "CLOSED"}:
            _check_repo_path(
                root, engineering.get("evidence"),
                "engineering_performance.evidence", prefix, errors)
            if verdict == "PLANNED":
                errors.append(
                    f"{prefix}: {effective_state} cannot retain a PLANNED engineering verdict")
        if tier in {"P1", "P2"}:
            margin = engineering.get("material_margin_pct")
            if not isinstance(margin, (int, float)) or margin <= 0:
                errors.append(
                    f"{prefix}: {tier} requires a positive material_margin_pct")
            machine = _require_string(
                engineering, "machine_profile", prefix, errors)
            if machine is not None and machine not in machine_profiles:
                errors.append(f"{prefix}: unknown engineering machine profile {machine!r}")
            if (machine is not None and isinstance(profiles, list) and
                    machine not in profiles):
                errors.append(
                    f"{prefix}: engineering machine profile {machine!r} is absent "
                    "from validation_machine_profiles")
        if tier == "P0" and mode != "STRUCTURAL":
            errors.append(f"{prefix}: P0 engineering mode must be STRUCTURAL")
        if tier == "P0" and instrument != "closure-correctness-v1":
            errors.append(
                f"{prefix}: P0 must use closure-correctness-v1")
        if tier == "P1" and mode != "P1_ENGINEERING":
            errors.append(f"{prefix}: P1 engineering mode must be P1_ENGINEERING")
        if tier == "P1" and instrument != "closure-perf-v1":
            errors.append(f"{prefix}: P1 must use closure-perf-v1")
        if tier == "P2" and mode != "P2_CONSUMER":
            errors.append(f"{prefix}: P2 engineering mode must be P2_CONSUMER")
        if tier == "P2" and instrument != "closure-perf-v1":
            errors.append(f"{prefix}: P2 must use closure-perf-v1")

    formal = record.get("formal_performance")
    if not isinstance(formal, dict):
        errors.append(f"{prefix}: formal_performance must be an object")
    else:
        formal_state = formal.get("state")
        if formal_state not in FORMAL_STATES:
            errors.append(f"{prefix}: invalid formal performance state {formal_state!r}")
        if formal_state != "NONE":
            formal_instrument = _require_string(
                formal, "instrument_profile", prefix, errors)
            if formal_instrument != "formal-release-v1":
                errors.append(
                    f"{prefix}: non-NONE Formal state must use formal-release-v1")
        if formal_state in {"CERTIFIED", "FAILED", "EXPIRED"}:
            _check_repo_path(
                root, formal.get("evidence"),
                "formal_performance.evidence", prefix, errors)

    if effective_state in {"CANDIDATE", "VERIFIED", "CLOSED"}:
        _check_commit(record, "candidate_commit", prefix, errors)
        _require_string(record, "candidate_tree", prefix, errors)
        tree = record.get("candidate_tree")
        if isinstance(tree, str) and COMMIT_RE.fullmatch(tree) is None:
            errors.append(f"{prefix}: candidate_tree is not a lowercase Git tree id")

    if effective_state in {"VERIFIED", "CLOSED"}:
        gates = record.get("gate_verdicts")
        if not isinstance(gates, dict):
            errors.append(f"{prefix}: gate_verdicts must be an object")
        else:
            missing = sorted(GATE_NAMES - set(gates))
            extra = sorted(set(gates) - GATE_NAMES)
            if missing:
                errors.append(f"{prefix}: missing gate verdicts: {', '.join(missing)}")
            if extra:
                errors.append(f"{prefix}: unknown gate verdicts: {', '.join(extra)}")
            for gate, verdict in gates.items():
                if verdict not in GATE_VERDICTS:
                    errors.append(f"{prefix}: invalid {gate} gate verdict {verdict!r}")

    if state in {"CLOSED", "BLOCKED"}:
        _check_commit(record, "end_commit", prefix, errors)
        _check_commit(record, "registry_anchor_commit", prefix, errors)
        _check_repo_path(
            root, record.get("closure_evidence"),
            "closure_evidence", prefix, errors)
        _check_nonempty_string_list(record, "known_limitations", prefix, errors)
        _check_nonempty_string_list(record, "reopen_triggers", prefix, errors)
        _require_string(record, "summary", prefix, errors)

        loc = record.get("loc")
        if not isinstance(loc, dict):
            errors.append(f"{prefix}: terminal round requires a loc object")
        else:
            _check_commit(loc, "snapshot_commit", prefix, errors)
            missing_loc = sorted(LOC_FIELDS - set(loc))
            if missing_loc:
                errors.append(f"{prefix}: missing LOC fields: {', '.join(missing_loc)}")
            for field in LOC_FIELDS:
                value = loc.get(field)
                if not isinstance(value, int) or value <= 0:
                    errors.append(f"{prefix}: LOC '{field}' must be a positive integer")

    if state == "CLOSED":
        if _contains_tbd(record):
            errors.append(f"{prefix}: CLOSED record contains TBD")
        gates = record.get("gate_verdicts")
        if isinstance(gates, dict):
            blockers = [
                gate for gate, verdict in gates.items()
                if verdict != "PASS"
            ]
            if blockers:
                errors.append(
                    f"{prefix}: CLOSED has non-passing gates: {', '.join(blockers)}")
            if gates.get("engineering_performance") != "PASS":
                errors.append(
                    f"{prefix}: CLOSED requires engineering_performance gate PASS")
        if isinstance(engineering, dict) and engineering.get("verdict") != "PASS":
            errors.append(f"{prefix}: CLOSED requires engineering verdict PASS")



def validate_registry(data: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != 1:
        errors.append("registry schema_version must be 1")
    if data.get("methodology") != "module-closure-v1":
        errors.append("registry methodology must be 'module-closure-v1'")
    updated = data.get("updated_at")
    if not isinstance(updated, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", updated) is None:
        errors.append("registry updated_at must be YYYY-MM-DD")
    else:
        try:
            datetime.date.fromisoformat(updated)
        except ValueError:
            errors.append("registry updated_at must be a valid calendar date")

    modules = data.get("modules")
    if not isinstance(modules, list):
        return errors + ["registry modules must be an array"]
    try:
        instrument_profiles, machine_profiles = declared_performance_profiles(root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
        instrument_profiles, machine_profiles = set(), set()
    for record in modules:
        validate_module(
            record, root, machine_profiles, instrument_profiles, errors)

    actual = [
        record.get("module") for record in modules
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    ]
    if len(actual) != len(set(actual)):
        errors.append("registry contains duplicate module ids")
    try:
        expected = declared_components(root)
    except (OSError, ValueError) as error:
        errors.append(str(error))
    else:
        if actual != expected:
            errors.append(
                "registry module order/content differs from HyleComponents.cmake: "
                f"registry={actual!r}, components={expected!r}")
    return errors


def validate_transitions(
        previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    before_order = [
        record["module"] for record in previous.get("modules", [])
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    ]
    after_order = [
        record["module"] for record in current.get("modules", [])
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    ]
    before = {
        record["module"]: record for record in previous.get("modules", [])
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    }
    after = {
        record["module"]: record for record in current.get("modules", [])
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    }
    removed = sorted(set(before) - set(after))
    if removed:
        errors.append(
            "transition comparison does not allow component removal or rename: "
            + ", ".join(removed))

    surviving_after = [component for component in after_order if component in before]
    if surviving_after != before_order:
        errors.append(
            "transition comparison does not allow reordering existing components")

    added = [component for component in after_order if component not in before]
    for component in added:
        if after[component].get("product_state") != "QUEUED":
            errors.append(
                f"new component {component!r} must enter as 'QUEUED', got "
                f"{after[component].get('product_state')!r}")

    for component in sorted(set(before) & set(after)):
        old_state = before[component].get("product_state")
        new_state = after[component].get("product_state")
        allowed = ALLOWED_TRANSITIONS.get(old_state, set())
        if new_state not in allowed:
            errors.append(
                f"module {component!r}: illegal transition {old_state!r} -> {new_state!r}")
    return errors


def _markdown_link(repo_path: str, label: str) -> str:
    relative = posixpath.relpath(repo_path, "docs/implementation")
    return f"[{label}]({relative})"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_status_table(data: dict[str, Any]) -> str:
    lines = [
        "| Component | Product | Tier | Engineering performance | Formal | Current record | Summary |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in data["modules"]:
        tier = record.get("performance_tier", "—")
        engineering = record.get("engineering_performance")
        if isinstance(engineering, dict):
            engineering_text = f"{engineering.get('verdict')} / {engineering.get('mode')}"
            margin = engineering.get("material_margin_pct")
            if isinstance(margin, (int, float)):
                engineering_text += f" / {margin:g}%"
        else:
            engineering_text = "—"
        formal = record.get("formal_performance")
        formal_text = formal.get("state", "—") if isinstance(formal, dict) else "—"
        task = record.get("closure_task")
        task_text = _markdown_link(task, "task") if isinstance(task, str) else "—"
        row = (
            record["module"],
            record["product_state"],
            tier,
            engineering_text,
            formal_text,
            task_text,
            record.get("summary", "—"),
        )
        lines.append("| " + " | ".join(_escape_cell(item) for item in row) + " |")
    return "\n".join(lines)


def validate_status_page(data: dict[str, Any], page_text: str) -> list[str]:
    errors: list[str] = []
    if page_text.count(STATUS_TABLE_BEGIN) != 1 or page_text.count(STATUS_TABLE_END) != 1:
        return ["status.md must contain exactly one generated module status marker pair"]
    start = page_text.index(STATUS_TABLE_BEGIN) + len(STATUS_TABLE_BEGIN)
    end = page_text.index(STATUS_TABLE_END)
    if end <= start:
        return ["status.md generated status markers are reversed"]
    actual = page_text[start:end].strip()
    expected = render_status_table(data)
    if actual != expected:
        errors.append(
            "status.md module table differs from module-status.json; "
            "run scripts/check-module-closure.py --print-table and update it")
    return errors


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--status-page", type=Path)
    parser.add_argument(
        "--previous", type=Path,
        help="optional previous registry used to reject skipped state transitions")
    parser.add_argument(
        "--print-table", action="store_true",
        help="print the canonical Markdown table instead of checking status.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    registry_path = args.registry or root / "docs" / "implementation" / "module-status.json"
    status_path = args.status_page or root / "docs" / "implementation" / "status.md"
    try:
        data = load_registry(registry_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"module closure contract failed: {error}", file=sys.stderr)
        return 1

    errors = validate_registry(data, root)
    if args.previous is not None:
        try:
            previous = load_registry(args.previous)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"cannot read previous registry: {error}")
        else:
            errors.extend(validate_transitions(previous, data))

    if args.print_table:
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            return 1
        print(render_status_table(data))
        return 0

    try:
        page_text = status_path.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"cannot read status page: {error}")
    else:
        errors.extend(validate_status_page(data, page_text))

    if errors:
        print("module closure contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print(
        f"module closure contract passed: {len(data['modules'])} components, "
        f"schema {data['schema_version']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
