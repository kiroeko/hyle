"""Shared child-environment sanitizer contract for performance evidence."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from typing import Any

from hyle_perf.model import ValidationError


PERF_ENVIRONMENT_BLOCKLIST = frozenset({
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
    "TRACY_ONLY_LOCALHOST",
    "UBSAN_OPTIONS",
})
PERF_ENVIRONMENT_PREFIXES = ("HYLE_BENCH_", "HYLE_PERF_")
PERF_VENDOR_RUNTIME_CONTROLS = {
    # Tracy otherwise listens on wildcard interfaces and can trigger one
    # Windows Firewall consent dialog per fresh benchmark artifact path.  The
    # value is deliberately exact: other Tracy switches can change event
    # availability or the measured enqueue path and require a new review.
    "TRACY_ONLY_LOCALHOST": frozenset({"1"}),
}
PERF_ENVIRONMENT_FIXED = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
_HEX_DIGEST_RE = re.compile(r"[0-9a-f]{64}")


def child_environment_policy_record(*, path_value: str | None = None) -> dict[str, Any]:
    """Record the fixed sanitizer policy without leaking the raw environment."""
    if path_value is None:
        path_value = os.environ.get("PATH", "")
    return {
        "schema": "hyle.perf.child-environment-policy/v1",
        "cleared_names": sorted(PERF_ENVIRONMENT_BLOCKLIST),
        "cleared_prefixes": list(PERF_ENVIRONMENT_PREFIXES),
        "fixed_values": dict(sorted(PERF_ENVIRONMENT_FIXED.items())),
        "path_sha256": hashlib.sha256(path_value.encode("utf-8")).hexdigest(),
        "raw_environment_recorded": False,
    }


def validate_child_environment_policy_record(value: Any) -> dict[str, Any]:
    """Reject self-reported or weakened sanitizer-policy records."""
    if not isinstance(value, Mapping):
        raise ValidationError("child environment policy must be an object")
    record = dict(value)
    expected = child_environment_policy_record(path_value="")
    if set(record) != set(expected):
        raise ValidationError("child environment policy has invalid fields")
    for key in ("schema", "cleared_names", "cleared_prefixes",
                "fixed_values", "raw_environment_recorded"):
        if record.get(key) != expected[key]:
            raise ValidationError(
                f"child environment policy differs at {key!r}")
    digest = record.get("path_sha256")
    if not isinstance(digest, str) or not _HEX_DIGEST_RE.fullmatch(digest):
        raise ValidationError("child environment policy PATH digest is invalid")
    return record
