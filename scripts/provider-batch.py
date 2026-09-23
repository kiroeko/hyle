#!/usr/bin/env python3
"""Plan mature-provider batches without duplicating validation semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


SCHEMA_VERSION = 1
ROUTES = ("cpu", "cuda", "apple", "vulkan")
SYSTEMS = ("Windows", "Linux", "Darwin")
DELIVERIES = ("source", "shared", "static")
GATE_KINDS = ("correctness", "resource", "delivery", "performance", "off")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
BATCH_RE = re.compile(r"^B[2-6]$")
REQUIREMENT_RE = re.compile(r"^F(?:0[5-9]|1[0-8])\.[1-9][0-9]*$")
CACHE_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class BatchError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    data = value if isinstance(value, bytes) else canonical_bytes(value)
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True,
        check=False)
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BatchError(f"git {' '.join(args)} failed: {detail}")
    return completed


def resolve_commit(root: Path, reference: str) -> str:
    commit = run_git(root, "rev-parse", "--verify", reference + "^{commit}").stdout.strip()
    if not HEX40_RE.fullmatch(commit):
        raise BatchError(f"Git returned an invalid commit for {reference!r}: {commit!r}")
    return commit


def commit_tree(root: Path, commit: str) -> str:
    tree = run_git(root, "rev-parse", commit + "^{tree}").stdout.strip()
    if not HEX40_RE.fullmatch(tree):
        raise BatchError(f"Git returned an invalid tree for {commit}: {tree!r}")
    return tree


def require_ancestor(root: Path, baseline: str, candidate: str) -> None:
    result = run_git(root, "merge-base", "--is-ancestor", baseline, candidate,
                     check=False)
    if result.returncode != 0:
        raise BatchError(f"baseline {baseline} is not an ancestor of candidate {candidate}")


def require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BatchError(f"{label} must be an object")
    return value


def require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BatchError(f"{label} must be an array")
    return value


def require_keys(value: dict[str, Any], *, required: set[str], optional: set[str],
                 label: str) -> None:
    missing = sorted(required - value.keys())
    extra = sorted(value.keys() - required - optional)
    if missing:
        raise BatchError(f"{label} is missing fields: {', '.join(missing)}")
    if extra:
        raise BatchError(f"{label} has unknown fields: {', '.join(extra)}")


def require_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise BatchError(f"{label} must match {ID_RE.pattern}")
    return value


def require_strings(value: Any, label: str, *, nonempty: bool = True) -> list[str]:
    rows = require_list(value, label)
    if nonempty and not rows:
        raise BatchError(f"{label} must not be empty")
    if any(not isinstance(row, str) or not row for row in rows):
        raise BatchError(f"{label} must contain non-empty strings")
    if len(rows) != len(set(rows)):
        raise BatchError(f"{label} contains duplicates")
    return rows


def validate_manifest(value: Any) -> dict[str, Any]:
    manifest = require_object(value, "manifest")
    require_keys(
        manifest,
        required={"schema_version", "id", "baseline", "heavy_parallelism",
                  "heavy_route_order", "batches", "provider_features", "profiles",
                  "gate_groups", "residuals"},
        optional={"description"}, label="manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise BatchError(f"manifest schema_version must be {SCHEMA_VERSION}")
    require_id(manifest["id"], "manifest.id")
    if not isinstance(manifest["baseline"], str) or not HEX40_RE.fullmatch(manifest["baseline"]):
        raise BatchError("manifest.baseline must be a full lowercase Git commit")
    if manifest["heavy_parallelism"] != 2:
        raise BatchError("manifest.heavy_parallelism must be 2")
    if manifest["heavy_route_order"] != list(ROUTES):
        raise BatchError("manifest.heavy_route_order must be cpu,cuda,apple,vulkan")

    batches: dict[str, set[str]] = {}
    all_requirements: set[str] = set()
    for index, raw in enumerate(require_list(manifest["batches"], "manifest.batches")):
        row = require_object(raw, f"manifest.batches[{index}]")
        require_keys(row, required={"id", "requirements", "status"},
                     optional={"description"}, label=f"batch[{index}]")
        batch_id = row["id"]
        if not isinstance(batch_id, str) or not BATCH_RE.fullmatch(batch_id):
            raise BatchError(f"batch[{index}].id must be B2 through B6")
        if batch_id in batches:
            raise BatchError(f"duplicate batch id {batch_id}")
        requirements = require_strings(row["requirements"], f"batch {batch_id} requirements")
        invalid = [item for item in requirements if not REQUIREMENT_RE.fullmatch(item)]
        if invalid:
            raise BatchError(f"batch {batch_id} has invalid requirements: {', '.join(invalid)}")
        overlap = all_requirements.intersection(requirements)
        if overlap:
            raise BatchError(f"requirements appear in multiple batches: {', '.join(sorted(overlap))}")
        if row["status"] not in {"READY_TO_SCOPE", "QUEUED_BY_PLAN", "AUDITED", "CLOSED"}:
            raise BatchError(f"batch {batch_id} has invalid status {row['status']!r}")
        batches[batch_id] = set(requirements)
        all_requirements.update(requirements)
    if set(batches) != {"B2", "B3", "B4", "B5", "B6"}:
        raise BatchError("manifest must declare exactly B2 through B6")

    provider_ids: set[str] = set()
    covered_batches: set[str] = set()
    for index, raw in enumerate(require_list(manifest["provider_features"],
                                             "manifest.provider_features")):
        row = require_object(raw, f"provider_features[{index}]")
        require_keys(row, required={"id", "pin", "build_features", "batches", "claim"},
                     optional={"notes"}, label=f"provider_features[{index}]")
        provider_id = require_id(row["id"], f"provider_features[{index}].id")
        if provider_id in provider_ids:
            raise BatchError(f"duplicate provider feature id {provider_id}")
        provider_ids.add(provider_id)
        if not isinstance(row["pin"], str) or not row["pin"]:
            raise BatchError(f"provider {provider_id} pin must be non-empty")
        require_strings(row["build_features"], f"provider {provider_id} build_features")
        provider_batches = require_strings(row["batches"], f"provider {provider_id} batches")
        unknown = set(provider_batches) - batches.keys()
        if unknown:
            raise BatchError(f"provider {provider_id} references unknown batches: {sorted(unknown)}")
        covered_batches.update(provider_batches)
        if row["claim"] != "COMPILED_ONLY":
            raise BatchError(f"provider {provider_id} claim must be COMPILED_ONLY")
    if covered_batches != set(batches):
        raise BatchError("provider feature superset must mention every batch B2 through B6")

    profile_ids: set[str] = set()
    profile_keys: set[tuple[str, str, str, str]] = set()
    delivery_by_route = {route: set() for route in ROUTES}
    source_profiles = 0
    for index, raw in enumerate(require_list(manifest["profiles"], "manifest.profiles")):
        row = require_object(raw, f"profiles[{index}]")
        require_keys(
            row,
            required={"id", "route", "system", "delivery", "factors", "cmake_cache",
                      "test_labels", "sanitizer", "toolchain"},
            optional={"notes"}, label=f"profiles[{index}]")
        profile_id = require_id(row["id"], f"profiles[{index}].id")
        if profile_id in profile_ids:
            raise BatchError(f"duplicate profile id {profile_id}")
        profile_ids.add(profile_id)
        route, system, delivery = row["route"], row["system"], row["delivery"]
        if route not in (*ROUTES, "all"):
            raise BatchError(f"profile {profile_id} has invalid route {route!r}")
        if system not in SYSTEMS:
            raise BatchError(f"profile {profile_id} has invalid system {system!r}")
        if delivery not in DELIVERIES:
            raise BatchError(f"profile {profile_id} has invalid delivery {delivery!r}")
        if not isinstance(row["sanitizer"], str) or not row["sanitizer"]:
            raise BatchError(f"profile {profile_id} sanitizer must be non-empty")
        if not isinstance(row["toolchain"], str) or not row["toolchain"]:
            raise BatchError(f"profile {profile_id} toolchain must be non-empty")
        key = (route, system, delivery, row["sanitizer"])
        if key in profile_keys:
            raise BatchError(f"duplicate route/system/delivery/sanitizer profile {key}")
        profile_keys.add(key)
        factors = require_strings(row["factors"], f"profile {profile_id} factors",
                                  nonempty=delivery != "source")
        if factors != sorted(factors):
            raise BatchError(f"profile {profile_id} factors must be sorted")
        require_strings(row["test_labels"], f"profile {profile_id} test_labels",
                        nonempty=delivery != "source")
        cache = require_object(row["cmake_cache"], f"profile {profile_id} cmake_cache")
        for cache_key, cache_value in cache.items():
            if not isinstance(cache_key, str) or not CACHE_RE.fullmatch(cache_key):
                raise BatchError(f"profile {profile_id} has invalid CMake key {cache_key!r}")
            if not isinstance(cache_value, str) or not cache_value:
                raise BatchError(f"profile {profile_id} CMake value {cache_key} must be non-empty")
        if delivery in {"static", "shared"}:
            expected = "ON" if delivery == "static" else "OFF"
            if cache.get("HYLE_STATIC") != expected:
                raise BatchError(f"profile {profile_id} must set HYLE_STATIC={expected}")
            if route == "all":
                raise BatchError(f"profile {profile_id} route all is only valid for source delivery")
            delivery_by_route[route].add(delivery)
        else:
            source_profiles += 1
            if route != "all":
                raise BatchError(f"source profile {profile_id} must use route all")
    if source_profiles != 1:
        raise BatchError("manifest must have exactly one source delivery profile")
    for route, deliveries in delivery_by_route.items():
        if deliveries != {"static", "shared"}:
            raise BatchError(f"route {route} must have both static and shared profiles")

    gate_ids: set[str] = set()
    for index, raw in enumerate(require_list(manifest["gate_groups"], "manifest.gate_groups")):
        row = require_object(raw, f"gate_groups[{index}]")
        require_keys(
            row,
            required={"id", "batch", "profile", "kind", "requirements",
                      "expected_selector_gates", "applicability", "reason"},
            optional=set(), label=f"gate_groups[{index}]")
        gate_id = require_id(row["id"], f"gate_groups[{index}].id")
        if gate_id in gate_ids:
            raise BatchError(f"duplicate gate group id {gate_id}")
        gate_ids.add(gate_id)
        if row["batch"] not in batches:
            raise BatchError(f"gate group {gate_id} references unknown batch")
        if row["profile"] not in profile_ids:
            raise BatchError(f"gate group {gate_id} references unknown profile")
        if row["kind"] not in GATE_KINDS:
            raise BatchError(f"gate group {gate_id} has invalid kind")
        requirements = require_strings(row["requirements"], f"gate group {gate_id} requirements")
        if not set(requirements) <= batches[row["batch"]]:
            raise BatchError(f"gate group {gate_id} requirements escape batch {row['batch']}")
        selector_gates = require_strings(
            row["expected_selector_gates"], f"gate group {gate_id} expected_selector_gates",
            nonempty=row["applicability"] == "required")
        if row["applicability"] not in {"required", "not_applicable"}:
            raise BatchError(f"gate group {gate_id} has invalid applicability")
        if row["applicability"] == "not_applicable" and selector_gates:
            raise BatchError(f"not-applicable gate group {gate_id} cannot expect selector gates")
        if not isinstance(row["reason"], str) or not row["reason"]:
            raise BatchError(f"gate group {gate_id} requires a reason")

    residual_ids: set[str] = set()
    for index, raw in enumerate(require_list(manifest["residuals"], "manifest.residuals")):
        row = require_object(raw, f"residuals[{index}]")
        require_keys(row, required={"id", "batch", "route", "requirements", "state",
                                    "reason", "owner"}, optional=set(),
                     label=f"residuals[{index}]")
        residual_id = require_id(row["id"], f"residuals[{index}].id")
        if residual_id in residual_ids:
            raise BatchError(f"duplicate residual id {residual_id}")
        residual_ids.add(residual_id)
        if row["batch"] not in batches or row["route"] not in ROUTES:
            raise BatchError(f"residual {residual_id} has invalid batch/route")
        requirements = require_strings(row["requirements"], f"residual {residual_id} requirements")
        if not set(requirements) <= batches[row["batch"]]:
            raise BatchError(f"residual {residual_id} requirements escape its batch")
        if row["state"] != "UNPROVEN":
            raise BatchError(f"residual {residual_id} state must be UNPROVEN")
        if any(not isinstance(row[field], str) or not row[field]
               for field in ("reason", "owner")):
            raise BatchError(f"residual {residual_id} reason/owner must be non-empty")
    return manifest


def load_manifest(path: Path) -> dict[str, Any]:
    return validate_manifest(read_json(path))


def manifest_digest(manifest: dict[str, Any]) -> str:
    """Hash manifest semantics so Git LF and Windows CRLF checkouts agree."""
    return digest(manifest)


def require_candidate_manifest(root: Path, manifest_path: Path, candidate: str,
                               manifest: dict[str, Any]) -> str:
    try:
        relative = manifest_path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise BatchError("manifest must be inside the canonical repository") from exc
    blob = run_git(root, "show", candidate + ":" + relative).stdout
    try:
        committed = validate_manifest(json.loads(blob))
    except json.JSONDecodeError as exc:
        raise BatchError(f"candidate manifest {relative} is not valid JSON: {exc}") from exc
    if manifest_digest(committed) != manifest_digest(manifest):
        raise BatchError("worktree manifest does not match the immutable candidate manifest")
    return relative


def profile_identity(profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "route": profile["route"],
        "delivery": profile["delivery"],
        "factors": profile["factors"],
        "cmake_cache": profile["cmake_cache"],
        "sanitizer": profile["sanitizer"],
        "toolchain": profile["toolchain"],
    }


def profile_commands(profile: dict[str, Any], parallelism: int) -> dict[str, Any]:
    build_id = profile["id"] + "-" + digest(profile_identity(profile))[:16]
    build = "{build_root}/" + build_id
    cache = dict(profile["cmake_cache"])
    configure = ["cmake", "-S", "{source}", "-B", build, "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release",
                 "-DHYLE_DEV_FACTORS=" + ",".join(profile["factors"]),
                 "-DHYLE_BUILD_TESTS=" + ("OFF" if profile["delivery"] == "source" else "ON"),
                 "-DHYLE_BUILD_BENCH=OFF", "-DHYLE_WERROR=ON"]
    configure.extend("-D" + key + "=" + cache[key] for key in sorted(cache))
    if profile["delivery"] == "source":
        build_command = ["cmake", "--build", build, "--target", "hyle_source_package",
                         "--parallel", str(parallelism)]
        tests: list[list[str]] = []
    else:
        build_command = ["cmake", "--build", build, "--parallel", str(parallelism)]
        labels = "^(" + "|".join(re.escape(item) for item in profile["test_labels"]) + ")$"
        tests = [["ctest", "--test-dir", build, "--output-on-failure",
                  "--no-tests=error", "--parallel", "1", "--timeout", "300", "-L", labels]]
    return {"build_id": build_id, "configure": configure, "build": build_command,
            "tests": tests}


def make_plan(root: Path, manifest_path: Path, candidate_ref: str) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    baseline = resolve_commit(root, manifest["baseline"])
    candidate = resolve_commit(root, candidate_ref)
    require_ancestor(root, baseline, candidate)
    manifest_relative = require_candidate_manifest(root, manifest_path, candidate, manifest)
    candidate_tree = commit_tree(root, candidate)
    route_order = {route: index for index, route in enumerate(ROUTES)}

    profiles = []
    by_id = {profile["id"]: profile for profile in manifest["profiles"]}
    ordered = sorted(
        manifest["profiles"],
        key=lambda row: (-1 if row["route"] == "all" else route_order[row["route"]],
                         0 if row["delivery"] == "shared" else 1, row["id"]))
    for profile in ordered:
        identity = profile_identity(profile)
        profiles.append({
            "id": profile["id"], "route": profile["route"],
            "system": profile["system"], "delivery": profile["delivery"],
            "build_identity": identity, "build_identity_sha256": digest(identity),
            "commands": profile_commands(profile, manifest["heavy_parallelism"]),
        })

    gate_groups = []
    for group in manifest["gate_groups"]:
        status = "SELECTED" if group["applicability"] == "required" else "SKIPPED"
        gate_groups.append({**group, "status": status,
                            "profile_build_identity_sha256": digest(profile_identity(by_id[group["profile"]]))})
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": manifest_relative,
        "manifest_sha256": manifest_digest(manifest),
        "batch_id": manifest["id"],
        "baseline_commit": baseline,
        "candidate_commit": candidate,
        "candidate_tree": candidate_tree,
        "heavy_parallelism": manifest["heavy_parallelism"],
        "heavy_route_order": manifest["heavy_route_order"],
        "profiles": profiles,
        "gate_groups": gate_groups,
        "residuals": manifest["residuals"],
        "verdict": "READY" if gate_groups else "PROFILES_READY",
    }


def create_bundle(root: Path, manifest_path: Path, candidate_ref: str,
                  output: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dirty = run_git(root, "status", "--porcelain", "--untracked-files=all").stdout
    if dirty:
        raise BatchError("bundle requires a clean canonical worktree")
    baseline = resolve_commit(root, manifest["baseline"])
    candidate = resolve_commit(root, candidate_ref)
    if baseline == candidate:
        raise BatchError("bundle has no commits after baseline")
    require_ancestor(root, baseline, candidate)
    require_candidate_manifest(root, manifest_path, candidate, manifest)
    output = output.resolve()
    sidecar = output.with_name(output.name + ".json")
    if output.exists() or sidecar.exists():
        raise BatchError(f"refusing existing bundle or sidecar: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_ref = "refs/hyle-batch/candidate"
    existing = run_git(root, "show-ref", "--verify", temporary_ref, check=False)
    if existing.returncode == 0:
        raise BatchError(f"temporary bundle ref already exists: {temporary_ref}")
    run_git(root, "update-ref", temporary_ref, candidate)
    try:
        run_git(root, "bundle", "create", str(output), temporary_ref, "^" + baseline)
        run_git(root, "bundle", "verify", str(output))
    except Exception:
        output.unlink(missing_ok=True)
        raise
    finally:
        run_git(root, "update-ref", "-d", temporary_ref, check=False)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "kind": "hyle-provider-batch-incremental-git-bundle",
        "manifest_sha256": manifest_digest(manifest),
        "baseline_commit": baseline,
        "candidate_commit": candidate,
        "candidate_tree": commit_tree(root, candidate),
        "prerequisites": [baseline],
        "head_ref": temporary_ref,
        "bundle": output.name,
        "bundle_sha256": digest(output.read_bytes()),
        "bundle_bytes": output.stat().st_size,
        "remote_contract": [
            "remote must already contain baseline_commit",
            "git bundle verify must pass",
            "fetch head_ref and verify FETCH_HEAD equals candidate_commit",
        ],
    }
    write_json(sidecar, receipt)
    return receipt


def gate_rows(selector_result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for key in ("selected_gates", "skipped_gates"):
        for raw in require_list(selector_result.get(key), f"selector result {key}"):
            row = require_object(raw, f"selector result {key} row")
            gate_id = row.get("id")
            if not isinstance(gate_id, str) or not gate_id:
                raise BatchError("selector result gate lacks id")
            if gate_id in rows:
                raise BatchError(f"selector result duplicates gate {gate_id}")
            rows[gate_id] = row
    return rows


def finalize(manifest_path: Path, plan_path: Path, ledger_path: Path) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    plan = require_object(read_json(plan_path), "plan")
    ledger = require_object(read_json(ledger_path), "ledger")
    require_keys(ledger, required={"schema_version", "candidate_commit", "candidate_tree", "runs"},
                 optional=set(), label="ledger")
    if ledger["schema_version"] != SCHEMA_VERSION:
        raise BatchError("ledger schema_version mismatch")
    if plan.get("manifest_sha256") != manifest_digest(manifest):
        raise BatchError("plan manifest digest does not match current manifest")
    for field in ("candidate_commit", "candidate_tree"):
        if ledger.get(field) != plan.get(field):
            raise BatchError(f"ledger {field} does not match plan")
    groups = {row["id"]: row for row in plan.get("gate_groups", [])}
    if not groups:
        raise BatchError("cannot finalize a manifest with no gate groups")
    runs: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(require_list(ledger["runs"], "ledger.runs")):
        row = require_object(raw, f"ledger.runs[{index}]")
        require_keys(row, required={"gate_group", "selector_result", "sha256"},
                     optional=set(), label=f"ledger.runs[{index}]")
        group_id = row["gate_group"]
        if group_id in runs:
            raise BatchError(f"ledger duplicates gate group {group_id}")
        if group_id not in groups:
            raise BatchError(f"ledger references unknown gate group {group_id}")
        if not isinstance(row["sha256"], str) or not HEX64_RE.fullmatch(row["sha256"]):
            raise BatchError(f"ledger gate group {group_id} has invalid sha256")
        runs[group_id] = row

    outcomes = []
    for group_id, group in groups.items():
        if group["status"] == "SKIPPED":
            if group_id in runs:
                raise BatchError(f"not-applicable gate group {group_id} must not have a run")
            outcomes.append({"id": group_id, "status": "SKIPPED", "reason": group["reason"]})
            continue
        run = runs.get(group_id)
        if run is None:
            raise BatchError(f"required gate group {group_id} has no selector result")
        result_path = (ledger_path.parent / run["selector_result"]).resolve()
        try:
            result_bytes = result_path.read_bytes()
        except OSError as exc:
            raise BatchError(f"cannot read selector result for {group_id}: {exc}") from exc
        if digest(result_bytes) != run["sha256"]:
            raise BatchError(f"selector result digest mismatch for {group_id}")
        result = require_object(json.loads(result_bytes.decode("utf-8")),
                                f"selector result {group_id}")
        if result.get("verdict") != "PASS":
            raise BatchError(f"selector result for {group_id} is not PASS")
        for field in ("candidate_commit", "candidate_tree"):
            if result.get(field) != plan.get(field):
                raise BatchError(f"selector result {field} mismatch for {group_id}")
        available = gate_rows(result)
        accepted = []
        for gate_id in group["expected_selector_gates"]:
            gate = available.get(gate_id)
            if gate is None:
                raise BatchError(f"selector result for {group_id} lacks gate {gate_id}")
            if gate.get("status") not in {"PASS", "REUSED"}:
                raise BatchError(
                    f"selector gate {gate_id} for {group_id} is {gate.get('status')}, not PASS/REUSED")
            accepted.append({"id": gate_id, "status": gate["status"],
                             "cache_key": gate.get("cache_key"),
                             "receipt": gate.get("receipt")})
        outcomes.append({"id": group_id, "status": "PASS",
                         "selector_result": run["selector_result"],
                         "selector_result_sha256": run["sha256"], "gates": accepted})
    unexpected = sorted(set(runs) - groups.keys())
    if unexpected:
        raise BatchError(f"ledger has unexpected runs: {', '.join(unexpected)}")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "hyle-provider-batch-terminal",
        "batch_id": manifest["id"],
        "candidate_commit": plan["candidate_commit"],
        "candidate_tree": plan["candidate_tree"],
        "manifest_sha256": plan["manifest_sha256"],
        "plan_sha256": digest(plan_path.read_bytes()),
        "ledger_sha256": digest(ledger_path.read_bytes()),
        "outcome": "PASS_WITH_RECORDED_RESIDUALS" if manifest["residuals"] else "PASS",
        "gate_groups": outcomes,
        "residuals": manifest["residuals"],
        "coverage_policy": "Only explicit requirement evidence may update the coverage matrix; residuals stay UNPROVEN.",
    }


def default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root())
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--manifest", type=Path, required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--candidate", default="HEAD")
    plan.add_argument("--output", type=Path, required=True)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--manifest", type=Path, required=True)
    bundle.add_argument("--candidate", default="HEAD")
    bundle.add_argument("--output", type=Path, required=True)
    terminal = sub.add_parser("finalize")
    terminal.add_argument("--manifest", type=Path, required=True)
    terminal.add_argument("--plan", type=Path, required=True)
    terminal.add_argument("--ledger", type=Path, required=True)
    terminal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "validate":
            manifest = load_manifest(args.manifest)
            print(json.dumps({"verdict": "PASS", "id": manifest["id"],
                              "profiles": len(manifest["profiles"]),
                              "gate_groups": len(manifest["gate_groups"])},
                             ensure_ascii=False))
        elif args.command == "plan":
            write_json(args.output, make_plan(root, args.manifest, args.candidate))
        elif args.command == "bundle":
            receipt = create_bundle(root, args.manifest, args.candidate, args.output)
            print(json.dumps(receipt, ensure_ascii=False))
        else:
            write_json(args.output, finalize(args.manifest, args.plan, args.ledger))
    except (BatchError, OSError, json.JSONDecodeError) as exc:
        print(f"provider batch error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
