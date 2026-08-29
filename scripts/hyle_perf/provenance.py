"""Reviewed, exact provenance-hash renewals for historical perf evidence.

The sampling manifest and frozen contract remain immutable.  A renewal only
allows the verifier to recognize that a later byte identity was reviewed as
non-material to one exact workload contract.  Unknown hashes, paths, workloads,
contracts, or edited review records continue to fail closed.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any, Mapping, Optional

from hyle_perf.model import (
    ValidationError,
    require_identifier,
    require_json_object,
    require_nonempty_string,
)


PROVENANCE_RENEWALS_SCHEMA = "hyle.perf.provenance-renewals/v1"
PROVENANCE_EXACT = "EXACT"
PROVENANCE_REVIEWED_NON_MATERIAL = "REVIEWED_NON_MATERIAL"
PROVENANCE_NOT_APPLICABLE = "NOT_APPLICABLE"
_HEX_DIGITS = frozenset("0123456789abcdef")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, field_name: str) -> str:
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in _HEX_DIGITS for character in value)):
        raise ValidationError(f"{field_name} is not a lowercase SHA-256")
    return value


def _require_repository_path(value: Any, field_name: str) -> str:
    text = require_nonempty_string(value, field_name)
    path = pathlib.PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or "\\" in text:
        raise ValidationError(
            f"{field_name} must be a repository-relative POSIX path")
    return text


def _strict_json_loads(payload: bytes, field_name: str) -> Any:
    def object_from_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(
                    f"{field_name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Any:
        raise ValidationError(
            f"{field_name} contains non-finite JSON constant {value!r}")

    try:
        return json.loads(
            payload.decode("utf-8"), object_pairs_hook=object_from_pairs,
            parse_constant=reject_constant)
    except ValidationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"{field_name} is not strict UTF-8 JSON: {error}") \
            from error


def validate_provenance_renewals(record: Mapping[str, Any]) -> dict[str, Any]:
    data = require_json_object(record, "provenance renewals")
    if set(data) != {"schema", "renewals"}:
        raise ValidationError("provenance renewals has invalid fields")
    if data["schema"] != PROVENANCE_RENEWALS_SCHEMA:
        raise ValidationError("provenance renewals schema is unsupported")
    raw_rows = data["renewals"]
    if not isinstance(raw_rows, list):
        raise ValidationError("provenance renewals.renewals must be an array")

    rows = []
    outgoing = set()
    for index, raw_row in enumerate(raw_rows):
        name = f"provenance renewals.renewals[{index}]"
        row = require_json_object(raw_row, name)
        fields = {
            "contract_id", "contract_sha256", "workload_id", "path",
            "from_sha256", "to_sha256", "decision", "scope",
            "review_task", "review_task_sha256", "reason",
            "reviewed_contract_facts",
        }
        if set(row) != fields:
            raise ValidationError(f"{name} has invalid fields")
        contract_id = require_identifier(
            row["contract_id"], name + ".contract_id")
        contract_sha256 = _require_sha256(
            row["contract_sha256"], name + ".contract_sha256")
        workload_id = require_identifier(
            row["workload_id"], name + ".workload_id")
        provenance_path = _require_repository_path(
            row["path"], name + ".path")
        from_sha256 = _require_sha256(
            row["from_sha256"], name + ".from_sha256")
        to_sha256 = _require_sha256(
            row["to_sha256"], name + ".to_sha256")
        if from_sha256 == to_sha256:
            raise ValidationError(f"{name} does not change provenance identity")
        if row["decision"] != "NON_MATERIAL":
            raise ValidationError(f"{name}.decision must be NON_MATERIAL")
        if row["scope"] != "ENGINEERING_REPLAY_ONLY":
            raise ValidationError(
                f"{name}.scope must be ENGINEERING_REPLAY_ONLY")
        review_task = _require_repository_path(
            row["review_task"], name + ".review_task")
        review_task_sha256 = _require_sha256(
            row["review_task_sha256"], name + ".review_task_sha256")
        reason = require_nonempty_string(row["reason"], name + ".reason")
        facts = row["reviewed_contract_facts"]
        if (not isinstance(facts, list) or not facts or
                any(not isinstance(fact, str) or not fact.strip()
                    for fact in facts)):
            raise ValidationError(
                f"{name}.reviewed_contract_facts must contain non-empty strings")

        edge_key = (
            contract_id, contract_sha256, workload_id, provenance_path,
            from_sha256,
        )
        if edge_key in outgoing:
            raise ValidationError(
                f"{name} creates a duplicate/branching renewal edge")
        outgoing.add(edge_key)
        rows.append({
            "contract_id": contract_id,
            "contract_sha256": contract_sha256,
            "workload_id": workload_id,
            "path": provenance_path,
            "from_sha256": from_sha256,
            "to_sha256": to_sha256,
            "decision": "NON_MATERIAL",
            "scope": "ENGINEERING_REPLAY_ONLY",
            "review_task": review_task,
            "review_task_sha256": review_task_sha256,
            "reason": reason,
            "reviewed_contract_facts": list(facts),
        })
    return {"schema": PROVENANCE_RENEWALS_SCHEMA, "renewals": rows}


def load_provenance_renewals(
        path: Optional[pathlib.Path]) -> tuple[dict[str, Any], Optional[str]]:
    if path is None:
        return {"schema": PROVENANCE_RENEWALS_SCHEMA, "renewals": []}, None
    path = path.resolve()
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValidationError(
            f"cannot read provenance renewal ledger {path}: {error}") from error
    record = require_json_object(
        _strict_json_loads(payload, f"provenance renewal ledger {path}"),
        f"provenance renewal ledger {path}")
    return validate_provenance_renewals(record), _sha256_bytes(payload)


def resolve_provenance_compatibility(
        renewals: Optional[Mapping[str, Any]], *, contract_id: str,
        contract_sha256: str, workload_id: str, provenance_path: str,
        frozen_sha256: str, current_sha256: str,
        repository_root: pathlib.Path,
        allow_reviewed_renewals: bool = True) -> dict[str, Any]:
    """Resolve one exact frozen->current hash chain and verify every review."""
    if frozen_sha256 == current_sha256:
        return {
            "workload_id": workload_id,
            "path": provenance_path,
            "compatibility": PROVENANCE_EXACT,
            "renewals": [],
        }
    if not allow_reviewed_renewals:
        raise ValidationError(
            "Formal provenance must remain byte-exact; reviewed engineering "
            "renewals are not accepted")

    ledger = validate_provenance_renewals(
        renewals or {"schema": PROVENANCE_RENEWALS_SCHEMA, "renewals": []})
    matching = [
        row for row in ledger["renewals"]
        if row["contract_id"] == contract_id and
        row["contract_sha256"] == contract_sha256 and
        row["workload_id"] == workload_id and
        row["path"] == provenance_path
    ]
    by_source = {row["from_sha256"]: row for row in matching}
    chain = []
    seen = set()
    cursor = frozen_sha256
    while cursor != current_sha256:
        if cursor in seen:
            raise ValidationError(
                "provenance renewal chain contains a cycle")
        seen.add(cursor)
        edge = by_source.get(cursor)
        if edge is None:
            raise ValidationError(
                "provenance file hash changed and no reviewed non-material "
                "renewal reaches the current hash")

        task_relative = pathlib.PurePosixPath(edge["review_task"])
        task_path = repository_root.joinpath(*task_relative.parts).resolve()
        try:
            task_path.relative_to(repository_root.resolve())
        except ValueError as error:
            raise ValidationError(
                "provenance renewal review task resolves outside repository") \
                from error
        if not task_path.is_file():
            raise ValidationError(
                f"provenance renewal review task is missing: {task_path}")
        if _sha256_file(task_path) != edge["review_task_sha256"]:
            raise ValidationError(
                "provenance renewal review task file hash changed")
        chain.append(edge)
        cursor = edge["to_sha256"]
        if len(chain) > len(matching):
            raise ValidationError(
                "provenance renewal chain cannot reach the current hash")

    return {
        "workload_id": workload_id,
        "path": provenance_path,
        "compatibility": PROVENANCE_REVIEWED_NON_MATERIAL,
        "renewals": [{
            "from_sha256": edge["from_sha256"],
            "to_sha256": edge["to_sha256"],
            "review_task": edge["review_task"],
            "review_task_sha256": edge["review_task_sha256"],
        } for edge in chain],
    }


def summarize_provenance_compatibility(
        records: list[Mapping[str, Any]]) -> str:
    if not records:
        return PROVENANCE_NOT_APPLICABLE
    if any(record.get("compatibility") == PROVENANCE_REVIEWED_NON_MATERIAL
           for record in records):
        return PROVENANCE_REVIEWED_NON_MATERIAL
    if all(record.get("compatibility") == PROVENANCE_EXACT
           for record in records):
        return PROVENANCE_EXACT
    raise ValidationError("provenance compatibility records are inconsistent")
