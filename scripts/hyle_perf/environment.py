"""Shared child-environment sanitizer contract for performance evidence."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from typing import Any

from hyle_perf.model import ValidationError


_CHILD_ENVIRONMENT_POLICY_V1 = "hyle.perf.child-environment-policy/v1"
_CHILD_ENVIRONMENT_POLICY_V2 = "hyle.perf.child-environment-policy/v2"
_CURRENT_CHILD_ENVIRONMENT_POLICY = _CHILD_ENVIRONMENT_POLICY_V2

# v1 was accidentally changed in place when Tracy's network-listener control
# was added.  Both byte shapes exist in retained evidence, so keep their exact
# fingerprints as legacy read-only inputs.  New evidence uses v2; future policy
# changes must add another immutable schema instead of editing these sets.
_PERF_ENVIRONMENT_BLOCKLIST_V1 = frozenset({
    "ASAN_OPTIONS",
    "CPUPROFILE",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "GLIBC_TUNABLES",
    "GOMP_CPU_AFFINITY",
    "KMP_AFFINITY",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LSAN_OPTIONS",
    "MALLOC_CONF",
    "MIMALLOC_OPTIONS",
    "TSAN_OPTIONS",
    "UBSAN_OPTIONS",
})
_PERF_ENVIRONMENT_BLOCKLIST_V2 = (
    _PERF_ENVIRONMENT_BLOCKLIST_V1 | {"TRACY_ONLY_LOCALHOST"})
_PERF_ENVIRONMENT_PREFIXES_V1 = ("HYLE_BENCH_", "HYLE_PERF_")
_PERF_ENVIRONMENT_FIXED_V1 = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}

PERF_ENVIRONMENT_BLOCKLIST = _PERF_ENVIRONMENT_BLOCKLIST_V2
PERF_ENVIRONMENT_PREFIXES = _PERF_ENVIRONMENT_PREFIXES_V1
PERF_VENDOR_RUNTIME_CONTROLS = {
    # Tracy otherwise listens on wildcard interfaces and can trigger one
    # Windows Firewall consent dialog per fresh benchmark artifact path.  The
    # value is deliberately exact: other Tracy switches can change event
    # availability or the measured enqueue path and require a new review.
    "TRACY_ONLY_LOCALHOST": frozenset({"1"}),
}
PERF_ENVIRONMENT_FIXED = _PERF_ENVIRONMENT_FIXED_V1
_HEX_DIGEST_RE = re.compile(r"[0-9a-f]{64}")

_POLICY_FIELDS = {
    "schema",
    "cleared_names",
    "cleared_prefixes",
    "fixed_values",
    "path_sha256",
    "raw_environment_recorded",
}


def _policy_shape(schema: str, cleared_names: frozenset[str]) -> dict[str, Any]:
    return {
        "schema": schema,
        "cleared_names": sorted(cleared_names),
        "cleared_prefixes": list(_PERF_ENVIRONMENT_PREFIXES_V1),
        "fixed_values": dict(sorted(_PERF_ENVIRONMENT_FIXED_V1.items())),
        "raw_environment_recorded": False,
    }


_KNOWN_POLICY_SHAPES = {
    # The second v1 shape is the short-lived in-place Tracy extension.  It is
    # accepted only for retained evidence; child_environment_policy_record()
    # never emits v1 again.
    _CHILD_ENVIRONMENT_POLICY_V1: (
        _policy_shape(_CHILD_ENVIRONMENT_POLICY_V1,
                      _PERF_ENVIRONMENT_BLOCKLIST_V1),
        _policy_shape(_CHILD_ENVIRONMENT_POLICY_V1,
                      _PERF_ENVIRONMENT_BLOCKLIST_V2),
    ),
    _CHILD_ENVIRONMENT_POLICY_V2: (
        _policy_shape(_CHILD_ENVIRONMENT_POLICY_V2,
                      _PERF_ENVIRONMENT_BLOCKLIST_V2),
    ),
}


def child_environment_policy_record(*, path_value: str | None = None) -> dict[str, Any]:
    """Record the fixed sanitizer policy without leaking the raw environment."""
    if path_value is None:
        path_value = os.environ.get("PATH", "")
    return {
        "schema": _CURRENT_CHILD_ENVIRONMENT_POLICY,
        "cleared_names": sorted(PERF_ENVIRONMENT_BLOCKLIST),
        "cleared_prefixes": list(PERF_ENVIRONMENT_PREFIXES),
        "fixed_values": dict(sorted(PERF_ENVIRONMENT_FIXED.items())),
        "path_sha256": hashlib.sha256(path_value.encode("utf-8")).hexdigest(),
        "raw_environment_recorded": False,
    }


def validate_child_environment_policy_record(value: Any) -> dict[str, Any]:
    """Accept exact known policy snapshots and reject weakened variants."""
    if not isinstance(value, Mapping):
        raise ValidationError("child environment policy must be an object")
    record = dict(value)
    if set(record) != _POLICY_FIELDS:
        raise ValidationError("child environment policy has invalid fields")
    schema = record.get("schema")
    known_shapes = _KNOWN_POLICY_SHAPES.get(schema)
    if known_shapes is None:
        raise ValidationError(
            f"child environment policy schema is unsupported: {schema!r}")
    shape = {key: record.get(key) for key in _POLICY_FIELDS
             if key != "path_sha256"}
    if not any(shape == expected for expected in known_shapes):
        raise ValidationError(
            f"child environment policy does not match a known {schema!r} "
            "snapshot")
    digest = record.get("path_sha256")
    if not isinstance(digest, str) or not _HEX_DIGEST_RE.fullmatch(digest):
        raise ValidationError("child environment policy PATH digest is invalid")
    return record


def child_environment_policy_compatibility(
        value: Any, contract: Mapping[str, Any]) -> str:
    """Prove whether a recorded policy is safe to replay under v2 rules.

    Deep replay validates the policy that actually governed the old child; it
    must not pretend today's policy was present.  The only legacy difference we
    know is Tracy's listener variable.  A v1 record without that clearing is
    compatible only when the frozen build contract explicitly disabled Tracy
    and no stratum attempted to restore the variable.
    """
    record = validate_child_environment_policy_record(value)
    cleared_names = frozenset(record["cleared_names"])
    if cleared_names == _PERF_ENVIRONMENT_BLOCKLIST_V2:
        return ("CURRENT" if record["schema"] ==
                _CURRENT_CHILD_ENVIRONMENT_POLICY else
                "LEGACY_SCHEMA_CURRENT_POLICY")

    runner = contract.get("extensions", {}).get("runner", {})
    cmake_cache = runner.get("cmake", {}).get("cache", {})
    trace_setting = str(cmake_cache.get("HYLE_TRACE", "")).strip().upper()
    controlled = set(runner.get("controlled_environment", []))
    strata_use_tracy_control = any(
        "TRACY_ONLY_LOCALHOST" in stratum.get("environment", {})
        for stratum in contract.get("strata", []))
    if (trace_setting in {"0", "FALSE", "OFF"} and
            "TRACY_ONLY_LOCALHOST" not in controlled and
            not strata_use_tracy_control):
        return "LEGACY_COMPATIBLE_TRACE_DISABLED"
    raise ValidationError(
        "legacy child environment policy did not clear "
        "TRACY_ONLY_LOCALHOST and the frozen contract does not prove Tracy "
        "was disabled")
