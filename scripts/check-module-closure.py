#!/usr/bin/env python3
"""Validate HYLE's current module registry and generated status view."""

from __future__ import annotations

import argparse
import functools
import json
import posixpath
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


STATES = {
    "QUEUED",
    "SCOPED",
    "AUDITED",
    "CANDIDATE",
    "CLOSED",
    "BLOCKED",
}
PERFORMANCE_TIERS = {"P0", "P1", "P2"}
ROOT_FIELDS = {"schema_version", "methodology", "modules"}
BASE_FIELDS = {"module", "state", "summary"}
STATUS_TABLE_BEGIN = "<!-- BEGIN GENERATED MODULE STATUS -->"
STATUS_TABLE_END = "<!-- END GENERATED MODULE STATUS -->"
OBJECT_ID_RE = re.compile(r"^[0-9a-f]{40}$")
COMPONENT_RE = re.compile(r"_hyle_declare_component\(\s*([a-z][a-z0-9_]*)")
COMPONENT_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")

ALLOWED_TRANSITIONS = {
    "QUEUED": {"QUEUED", "SCOPED"},
    "SCOPED": {"SCOPED", "AUDITED", "BLOCKED"},
    "AUDITED": {"AUDITED", "CANDIDATE", "BLOCKED"},
    "CANDIDATE": {"CANDIDATE", "AUDITED", "CLOSED", "BLOCKED"},
    "CLOSED": {"CLOSED", "SCOPED"},
    "BLOCKED": {"BLOCKED", "SCOPED"},
}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("registry root must be a JSON object")
    return value


def load_registry_ref(root: Path, revision: str) -> dict[str, Any]:
    if not revision or revision.startswith("-"):
        raise ValueError("previous revision must be a non-option Git revision")
    result = subprocess.run(
        ["git", "show", f"{revision}:docs/implementation/module-status.json"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"cannot read module registry at {revision!r}: {detail}")
    value = json.loads(result.stdout.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("previous registry root must be a JSON object")
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


def _require_string(
    record: dict[str, Any], field: str, prefix: str, errors: list[str]
) -> str | None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}: '{field}' must be a non-empty string")
        return None
    return value


def _check_repo_file(
    root: Path, value: Any, field: str, prefix: str, errors: list[str]
) -> None:
    if not isinstance(value, str) or not value:
        errors.append(f"{prefix}: '{field}' must be a non-empty repository file")
        return
    path = Path(value)
    if path.is_absolute() or re.match(r"^[A-Za-z]:", value):
        errors.append(f"{prefix}: '{field}' must be repository-relative: {value}")
        return
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        errors.append(f"{prefix}: '{field}' escapes the repository: {value}")
        return
    if not resolved.is_file():
        errors.append(f"{prefix}: '{field}' is not a repository file: {value}")


def _check_object_id(value: Any, field: str, prefix: str, errors: list[str]) -> None:
    if not isinstance(value, str) or OBJECT_ID_RE.fullmatch(value) is None:
        errors.append(f"{prefix}: '{field}' must be a full lowercase 40-digit Git id")


def _contains_tbd(value: Any) -> bool:
    if isinstance(value, str):
        return "TBD" in value.upper()
    if isinstance(value, list):
        return any(_contains_tbd(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_tbd(item) for item in value.values())
    return False


def _record_state(record: dict[str, Any]) -> Any:
    return record.get("state")


@functools.lru_cache(maxsize=None)
def _git_commit_reachable(root: str, commit: str) -> bool:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=root,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def _validate_candidate(
    value: Any, root: Path, prefix: str, errors: list[str]
) -> None:
    _check_object_id(value, "candidate", prefix, errors)
    if isinstance(value, str) and OBJECT_ID_RE.fullmatch(value):
        if not _git_commit_reachable(str(root.resolve()), value):
            errors.append(f"{prefix}: candidate is not a reachable Git commit: {value}")


def _validate_performance(
    value: Any, root: Path, prefix: str, errors: list[str]
) -> None:
    if not isinstance(value, dict):
        errors.append(f"{prefix}: performance must be an object")
        return
    allowed = {"tier", "evidence", "material_margin_pct"}
    extra = sorted(set(value) - allowed)
    if extra:
        errors.append(f"{prefix}: unknown performance fields: {', '.join(extra)}")
    tier = value.get("tier")
    if tier not in PERFORMANCE_TIERS:
        errors.append(f"{prefix}: invalid performance tier {tier!r}")
    _check_repo_file(root, value.get("evidence"), "performance.evidence", prefix, errors)
    margin = value.get("material_margin_pct")
    if tier in {"P1", "P2"}:
        if not isinstance(margin, (int, float)) or isinstance(margin, bool) or margin <= 0:
            errors.append(f"{prefix}: {tier} requires a positive material_margin_pct")
    elif "material_margin_pct" in value:
        errors.append(f"{prefix}: P0 must not declare a timing material margin")


def validate_module(record: Any, root: Path, errors: list[str]) -> None:
    if not isinstance(record, dict):
        errors.append("module entry must be a JSON object")
        return
    component = record.get("module")
    prefix = f"module {component!r}"
    if not isinstance(component, str) or COMPONENT_ID_RE.fullmatch(component) is None:
        errors.append("module entry has an invalid 'module' id")
        return
    state = record.get("state")
    if state not in STATES:
        errors.append(f"{prefix}: invalid state {state!r}")
        return
    _require_string(record, "summary", prefix, errors)

    allowed = set(BASE_FIELDS)
    if state != "QUEUED":
        allowed.add("record")
        _check_repo_file(root, record.get("record"), "record", prefix, errors)
    if state in {"CANDIDATE", "CLOSED"}:
        allowed.add("candidate")
        _validate_candidate(record.get("candidate"), root, prefix, errors)
    if state == "CLOSED":
        allowed.add("performance")
        _validate_performance(record.get("performance"), root, prefix, errors)
        if _contains_tbd(record):
            errors.append(f"{prefix}: CLOSED record contains TBD")
    elif state == "BLOCKED":
        allowed |= {
            "blocked_from",
            "blocker",
            "blocker_owner",
            "resume_condition",
            "candidate",
        }
        if record.get("blocked_from") not in {"SCOPED", "AUDITED", "CANDIDATE"}:
            errors.append(f"{prefix}: BLOCKED requires a valid blocked_from phase")
        for field in ("blocker", "blocker_owner", "resume_condition"):
            _require_string(record, field, prefix, errors)
        if "candidate" in record:
            _validate_candidate(record["candidate"], root, prefix, errors)

    extra = sorted(set(record) - allowed)
    if extra:
        errors.append(f"{prefix}: fields not valid for {state}: {', '.join(extra)}")


def validate_registry(data: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    if set(data) != ROOT_FIELDS:
        missing = sorted(ROOT_FIELDS - set(data))
        extra = sorted(set(data) - ROOT_FIELDS)
        if missing:
            errors.append(f"registry missing root fields: {', '.join(missing)}")
        if extra:
            errors.append(f"registry has retired or unknown root fields: {', '.join(extra)}")
    if data.get("schema_version") != 4:
        errors.append("registry schema_version must be 4")
    if data.get("methodology") != "module-closure-v4":
        errors.append("registry methodology must be 'module-closure-v4'")

    modules = data.get("modules")
    if not isinstance(modules, list):
        return errors + ["registry modules must be an array"]
    for record in modules:
        validate_module(record, root, errors)
    actual = [
        record.get("module")
        for record in modules
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
                f"registry={actual!r}, components={expected!r}"
            )
    return errors


def validate_transitions(
    previous: dict[str, Any], current: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    before_records = previous.get("modules", [])
    after_records = current.get("modules", [])
    if not isinstance(before_records, list) or not isinstance(after_records, list):
        return ["transition comparison requires modules arrays"]
    before_order = [
        record["module"]
        for record in before_records
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    ]
    after_order = [
        record["module"]
        for record in after_records
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    ]
    before = {
        record["module"]: record
        for record in before_records
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    }
    after = {
        record["module"]: record
        for record in after_records
        if isinstance(record, dict) and isinstance(record.get("module"), str)
    }
    removed = [
        component for component in before_order
        if component not in after and _record_state(before[component]) != "QUEUED"
    ]
    if removed:
        errors.append(
            "transition comparison does not allow component removal or rename: "
            + ", ".join(removed)
        )
    surviving_after = [component for component in after_order if component in before]
    surviving_before = [component for component in before_order if component in after]
    if surviving_after != surviving_before:
        errors.append("transition comparison does not allow reordering existing components")
    for component in after_order:
        if component not in before and _record_state(after[component]) != "QUEUED":
            errors.append(
                f"new component {component!r} must enter as 'QUEUED', got "
                f"{_record_state(after[component])!r}"
            )
    for component in before_order:
        if component not in after:
            continue
        old_state = _record_state(before[component])
        new_state = _record_state(after[component])
        if new_state not in ALLOWED_TRANSITIONS.get(old_state, set()):
            errors.append(
                f"module {component!r}: illegal transition {old_state!r} -> {new_state!r}"
            )
        if old_state == "CLOSED" and new_state == "SCOPED":
            if before[component].get("record") == after[component].get("record"):
                errors.append(
                    f"module {component!r}: CLOSED -> SCOPED requires a new closure record"
                )
    return errors


def _git_lines(root: Path, *args: str) -> list[str]:
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    if result.returncode:
        detail = result.stderr.decode(errors="replace").strip()
        raise ValueError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.decode().splitlines()


def validate_transition_history(
    root: Path, previous_ref: str, current: dict[str, Any]
) -> list[str]:
    """Validate each committed transition, then any current worktree transition."""
    head = _git_lines(root, "rev-parse", "--verify", "HEAD^{commit}")[0]
    start = _git_lines(root, "merge-base", previous_ref, head)[0]
    commits = _git_lines(
        root,
        "rev-list",
        "--reverse",
        "--topo-order",
        f"{start}..{head}",
        "--",
        "docs/implementation/module-status.json",
    )
    errors: list[str] = []
    for commit in commits:
        family = _git_lines(root, "rev-list", "--parents", "-n", "1", commit)[0].split()
        parents = family[1:]
        after = load_registry_ref(root, commit)
        if len(parents) == 1:
            for error in validate_transitions(load_registry_ref(root, parents[0]), after):
                errors.append(f"{commit[:12]}: {error}")
            continue
        if len(parents) > 1:
            parent_modules = []
            for parent in parents:
                registry = load_registry_ref(root, parent)
                parent_modules.append({
                    row.get("module"): row
                    for row in registry.get("modules", [])
                    if isinstance(row, dict) and isinstance(row.get("module"), str)
                })
            after_modules = {
                row.get("module"): row
                for row in after.get("modules", [])
                if isinstance(row, dict) and isinstance(row.get("module"), str)
            }
            all_names = set(after_modules)
            all_names.update(name for rows in parent_modules for name in rows)
            novel = [
                name for name in sorted(all_names)
                if not any(after_modules.get(name) == rows.get(name) for rows in parent_modules)
            ]
            if novel:
                errors.append(
                    f"{commit[:12]}: merge introduces novel registry records for "
                    f"{', '.join(novel)}; make the state transition in a follow-up commit"
                )

    head_registry = load_registry_ref(root, head)
    if head_registry != current:
        errors.extend(validate_transitions(head_registry, current))
    return errors


def _markdown_link(repo_path: str, label: str) -> str:
    relative = posixpath.relpath(repo_path, "docs/implementation")
    return f"[{label}]({relative})"


def _escape_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_status_table(data: dict[str, Any]) -> str:
    lines = [
        "| Component | State | Engineering performance | Current record | Summary |",
        "|---|---|---|---|---|",
    ]
    for record in data["modules"]:
        performance = record.get("performance")
        if isinstance(performance, dict):
            verdict = "PASS"
            evidence = performance.get("evidence")
            if isinstance(evidence, str):
                verdict = _markdown_link(evidence, verdict)
            performance_text = f"{performance.get('tier', '—')} / {verdict}"
            margin = performance.get("material_margin_pct")
            if isinstance(margin, (int, float)):
                performance_text += f" / {margin:g}%"
        else:
            performance_text = "—"
        task = record.get("record")
        task_text = _markdown_link(task, "record") if isinstance(task, str) else "—"
        row = (
            record["module"],
            record["state"],
            performance_text,
            task_text,
            record.get("summary", "—"),
        )
        lines.append("| " + " | ".join(_escape_cell(item) for item in row) + " |")
    return "\n".join(lines)


def _status_bounds(page_text: str) -> tuple[int, int]:
    if page_text.count(STATUS_TABLE_BEGIN) != 1 or page_text.count(STATUS_TABLE_END) != 1:
        raise ValueError("status.md must contain exactly one generated status marker pair")
    start = page_text.index(STATUS_TABLE_BEGIN) + len(STATUS_TABLE_BEGIN)
    end = page_text.index(STATUS_TABLE_END)
    if end <= start:
        raise ValueError("status.md generated status markers are reversed")
    return start, end


def validate_status_page(data: dict[str, Any], page_text: str) -> list[str]:
    try:
        start, end = _status_bounds(page_text)
    except ValueError as error:
        return [str(error)]
    if page_text[start:end].strip() != render_status_table(data):
        return [
            "status.md module table differs from module-status.json; run "
            "scripts/check-module-closure.py --write-status"
        ]
    return []


def write_status_page(data: dict[str, Any], path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = _status_bounds(text)
    replacement = "\n\n" + render_status_table(data) + "\n\n"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text[:start] + replacement + text[end:])


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repository_root())
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--status-page", type=Path)
    parser.add_argument(
        "--previous-ref",
        help="Git revision whose registry is compared for legal state transitions",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--print-table", action="store_true", help="print the generated Markdown table"
    )
    output.add_argument(
        "--write-status", action="store_true", help="update only the generated status table"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    root = args.root.resolve()
    registry_path = args.registry or root / "docs/implementation/module-status.json"
    status_path = args.status_page or root / "docs/implementation/status.md"
    try:
        data = load_registry(registry_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"module closure contract failed: {error}", file=sys.stderr)
        return 1

    errors = validate_registry(data, root)
    if args.previous_ref is not None:
        try:
            errors.extend(validate_transition_history(root, args.previous_ref, data))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            errors.append(f"cannot read previous registry revision: {error}")
    if errors:
        print("module closure contract failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if args.print_table:
        print(render_status_table(data))
        return 0
    if args.write_status:
        try:
            write_status_page(data, status_path)
        except (OSError, ValueError) as error:
            print(f"module closure contract failed: {error}", file=sys.stderr)
            return 1
        print(f"updated generated module status: {status_path}")
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
        f"schema {data['schema_version']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
