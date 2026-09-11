#!/usr/bin/env python3
"""Validate the current whole-library performance scorecard.

The scorecard is a snapshot of evidence, not a second module lifecycle registry.
This check deliberately keeps the invariant small: every currently CLOSED
component is bound to its current candidate and at least one claim, queued work
is excluded, workload identities are unique, and every claim carries the fields
needed to separate comparability, qualification, direction, and product action.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "docs/implementation/module-status.json"
SCORECARD_PATH = (
    REPO_ROOT
    / "docs/implementation/evidence/performance/full-library-scorecard.json"
)

KINDS = {"P0_STRUCTURAL", "P1_WORKLOAD", "ROUTE_POLICY", "P2_SLICE"}
COMPARISONS = {
    "SAME_CONTRACT_EXTERNAL",
    "SAME_CONTRACT_INTERNAL",
    "PROVIDER_SELECTION",
    "NATURAL_ROUTE",
    "NO_COMPARABLE_PEER",
}
QUALIFICATIONS = {
    "QUALIFIED_RETAINED",
    "QUALIFIED_HASHED_EXTERNAL",
    "CURRENT_DOCUMENT_ONLY",
    "STRUCTURAL",
    "REJECTED_DIAGNOSTIC",
}
DIRECTIONS = {"WIN", "PARITY", "INCONCLUSIVE", "LOSS", "N/A"}
PRACTICAL_RESULTS = {"PASS", "FAIL", "INCONCLUSIVE", "N/A"}
ACTIONS = {"KEEP", "INVESTIGATE", "REOPEN", "ACCEPT_TRADEOFF", "N/A"}
WEIGHTS = {"HIGH", "MEDIUM", "LOW"}
BINDINGS = {"EXACT", "INHERITED_UNCHANGED_ROUTE", "STRUCTURAL"}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be an object")
    return value


def require_text(record: dict[str, Any], field: str, prefix: str, errors: list[str]) -> None:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{prefix}: {field!r} must be non-empty text")


def main() -> int:
    errors: list[str] = []
    registry = load_json(REGISTRY_PATH)
    scorecard = load_json(SCORECARD_PATH)

    if scorecard.get("schema") != "hyle.performance.scorecard/v1":
        errors.append("scorecard: unsupported schema")

    registry_modules = {
        item["module"]: item
        for item in registry.get("modules", [])
        if isinstance(item, dict) and isinstance(item.get("module"), str)
    }
    closed = {name for name, item in registry_modules.items() if item.get("state") == "CLOSED"}
    not_closed = set(registry_modules) - closed

    coverage_items = scorecard.get("modules")
    if not isinstance(coverage_items, list):
        errors.append("scorecard: 'modules' must be an array")
        coverage_items = []
    coverage: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(coverage_items):
        prefix = f"modules[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        name = item.get("module")
        if not isinstance(name, str) or not name:
            errors.append(f"{prefix}: missing module")
            continue
        if name in coverage:
            errors.append(f"{prefix}: duplicate module {name!r}")
            continue
        coverage[name] = item
        current = registry_modules.get(name)
        if current is None:
            errors.append(f"{prefix}: unknown registry module {name!r}")
            continue
        if current.get("state") != "CLOSED":
            errors.append(f"{prefix}: {name!r} is not currently CLOSED")
        if item.get("candidate") != current.get("candidate"):
            errors.append(f"{prefix}: candidate does not match module-status.json")
        claims = item.get("claims")
        if not isinstance(claims, list) or not claims or not all(
            isinstance(claim, str) and claim for claim in claims
        ):
            errors.append(f"{prefix}: claims must be a non-empty string array")

    missing = sorted(closed - set(coverage))
    extra = sorted(set(coverage) - closed)
    if missing:
        errors.append(f"scorecard: missing CLOSED modules: {', '.join(missing)}")
    if extra:
        errors.append(f"scorecard: non-CLOSED modules included: {', '.join(extra)}")

    exclusions = scorecard.get("excluded_modules")
    if not isinstance(exclusions, list) or set(exclusions) != not_closed:
        errors.append(
            "scorecard: excluded_modules must exactly equal current non-CLOSED modules: "
            + ", ".join(sorted(not_closed))
        )

    claim_items = scorecard.get("claims")
    if not isinstance(claim_items, list):
        errors.append("scorecard: 'claims' must be an array")
        claim_items = []
    claims: dict[str, dict[str, Any]] = {}
    enum_fields = {
        "kind": KINDS,
        "comparison": COMPARISONS,
        "qualification": QUALIFICATIONS,
        "direction": DIRECTIONS,
        "practical_result": PRACTICAL_RESULTS,
        "engineering_action": ACTIONS,
        "consumer_weight": WEIGHTS,
        "candidate_binding": BINDINGS,
    }
    text_fields = (
        "consumer_shape",
        "contract",
        "peer",
        "scale",
        "absolute",
        "ratio_ci",
        "sensitivity",
        "mechanism",
        "limitations",
    )
    for index, item in enumerate(claim_items):
        prefix = f"claims[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        claim_id = item.get("id")
        if not isinstance(claim_id, str) or not claim_id:
            errors.append(f"{prefix}: missing id")
            continue
        if claim_id in claims:
            errors.append(f"{prefix}: duplicate id {claim_id!r}")
            continue
        claims[claim_id] = item
        modules = item.get("modules")
        if not isinstance(modules, list) or not modules or not all(
            isinstance(module, str) and module in closed for module in modules
        ):
            errors.append(f"{prefix}: modules must name one or more CLOSED components")
        for field, allowed in enum_fields.items():
            if item.get(field) not in allowed:
                errors.append(f"{prefix}: invalid {field} {item.get(field)!r}")
        for field in text_fields:
            require_text(item, field, prefix, errors)
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{prefix}: evidence must be a non-empty array")
        else:
            for value in evidence:
                if not isinstance(value, str) or not value:
                    errors.append(f"{prefix}: invalid evidence path {value!r}")
                    continue
                path_text = value.split("#", 1)[0]
                if not (REPO_ROOT / path_text).is_file():
                    errors.append(f"{prefix}: evidence file does not exist: {path_text}")
        if item.get("direction") == "LOSS":
            if item.get("engineering_action") not in {"REOPEN", "ACCEPT_TRADEOFF"}:
                errors.append(f"{prefix}: LOSS requires REOPEN or ACCEPT_TRADEOFF")
        if item.get("direction") == "PARITY":
            require_text(item, "parity_basis", prefix, errors)
        if item.get("comparison") == "NO_COMPARABLE_PEER" and item.get("direction") != "N/A":
            errors.append(f"{prefix}: no-peer claim direction must be N/A")

    for module, item in coverage.items():
        for claim_id in item.get("claims", []):
            claim = claims.get(claim_id)
            if claim is None:
                errors.append(f"module {module!r}: unknown claim {claim_id!r}")
            elif module not in claim.get("modules", []):
                errors.append(f"module {module!r}: claim {claim_id!r} is not reciprocal")
    for claim_id, item in claims.items():
        for module in item.get("modules", []):
            if claim_id not in coverage.get(module, {}).get("claims", []):
                errors.append(f"claim {claim_id!r}: module {module!r} is not reciprocal")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        "performance scorecard passed: "
        f"{len(coverage)} CLOSED modules, {len(claims)} independent claims, "
        f"{len(exclusions)} excluded modules"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
