#!/usr/bin/env python3
"""Machine registry and host probe shared by HYLE performance workflows.

This module deliberately does not contain benchmark statistics.  It answers
only whether the current execution environment matches a named machine
profile and records the host controls needed to interpret performance data.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import datetime as dt
import hashlib
import json
import os
import pathlib
import platform
import plistlib
import re
import secrets
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any


SCHEMA = "hyle-perf-machine-registry-v1"
PROBE_SCHEMA = "hyle-perf-machine-probe-v1"
MUTEX_SCHEMA = "hyle-perf-sampling-mutex-v1"
HOST_IDENTITY_SCHEMA = "hyle-perf-physical-host-identity-v1"
DEFAULT_REGISTRY_PATH = pathlib.Path(__file__).resolve().parents[1] / "perf-machines.json"

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]

_POWER_GUID_RE = re.compile(
    r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)
_REMOTE_FILESYSTEMS = {
    "9p",
    "cifs",
    "drvfs",
    "fuse.sshfs",
    "nfs",
    "nfs4",
    "smbfs",
}
_QOS_NAMES = {
    0x21: "interactive",
    0x19: "user-initiated",
    0x15: "default",
    0x11: "utility",
    0x09: "background",
    0x00: "unspecified",
}


class SamplingMutexError(RuntimeError):
    """Base error for the persistent cross-environment sampling mutex."""


class SamplingMutexBusy(SamplingMutexError):
    """Raised when another or stale sampling marker already exists."""

    def __init__(self, path: pathlib.Path, existing: Mapping[str, Any]):
        self.path = path
        self.existing = dict(existing)
        super().__init__(
            f"sampling mutex is busy at {path}; inspect both execution "
            "environments and recover the exact marker manually"
        )


class SamplingMutexOwnershipError(SamplingMutexError):
    """Raised when a lease refuses to remove a missing or foreign marker."""


def _safe_existing_marker(path: pathlib.Path) -> dict[str, Any]:
    """Read only non-sensitive, known fields from an existing marker."""
    try:
        data = path.read_bytes()
        if len(data) > 16 * 1024:
            raise ValueError("marker is too large")
        payload = json.loads(data.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("marker is not an object")
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
        return {"valid": False, "reason": type(error).__name__}
    allowed = (
        "schema",
        "mutex_key",
        "physical_host_id",
        "profile_id",
        "environment",
        "pid",
        "acquired_at_utc",
    )
    return {"valid": True, **{key: payload.get(key) for key in allowed}}


class SamplingMutex:
    """Persistent fail-closed marker shared by performance samplers."""

    def __init__(
        self,
        *,
        path: pathlib.Path,
        mutex_key: str,
        physical_host_id: str,
        profile_id: str,
        environment: str,
    ) -> None:
        self.path = path
        self.mutex_key = mutex_key
        self.physical_host_id = physical_host_id
        self.profile_id = profile_id
        self.environment = environment
        self._owner_token = secrets.token_hex(32)
        self._marker: dict[str, Any] | None = None

    @property
    def held(self) -> bool:
        return self._marker is not None

    def acquire(self) -> "SamplingMutex":
        if self.held:
            raise SamplingMutexError("sampling mutex lease is already held")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        marker = {
            "schema": MUTEX_SCHEMA,
            "mutex_key": self.mutex_key,
            "physical_host_id": self.physical_host_id,
            "profile_id": self.profile_id,
            "environment": self.environment,
            "pid": os.getpid(),
            "acquired_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "owner_token": self._owner_token,
        }
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except FileExistsError as error:
            raise SamplingMutexBusy(
                self.path, _safe_existing_marker(self.path)
            ) from error
        try:
            encoded = (json.dumps(marker, sort_keys=True) + "\n").encode("utf-8")
            offset = 0
            while offset < len(encoded):
                offset += os.write(descriptor, encoded[offset:])
            os.fsync(descriptor)
        except BaseException:
            # The newly created marker intentionally remains fail-closed.  A
            # partial write is indistinguishable from a crash and is recovered
            # only by the documented manual procedure.
            raise
        finally:
            os.close(descriptor)
        self._marker = marker
        return self

    def manifest_record(self) -> dict[str, Any]:
        if self._marker is None:
            raise SamplingMutexError("sampling mutex lease is not held")
        return {
            "schema": MUTEX_SCHEMA,
            "mutex_key": self.mutex_key,
            "physical_host_id": self.physical_host_id,
            "profile_id": self.profile_id,
            "environment": self.environment,
            "path": str(self.path),
            "pid": self._marker["pid"],
            "acquired_at_utc": self._marker["acquired_at_utc"],
            "owner_token_sha256": _sha256_text(self._owner_token),
            "raw_owner_token_recorded": False,
            "release_policy": "owner-token-match-only",
        }

    def release(self) -> None:
        if self._marker is None:
            raise SamplingMutexError("sampling mutex lease is not held")
        try:
            before = self.path.stat()
            data = self.path.read_bytes()
            if len(data) > 16 * 1024:
                raise ValueError("marker is too large")
            observed = json.loads(data.decode("utf-8"))
            after = self.path.stat()
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise SamplingMutexOwnershipError(
                f"sampling marker at {self.path} is missing or unreadable; "
                "refusing automatic removal"
            ) from error
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after:
            raise SamplingMutexOwnershipError(
                "sampling marker changed while ownership was checked; "
                "refusing automatic removal"
            )
        if (
            not isinstance(observed, dict)
            or observed.get("schema") != MUTEX_SCHEMA
            or observed.get("mutex_key") != self.mutex_key
            or observed.get("physical_host_id") != self.physical_host_id
            or observed.get("profile_id") != self.profile_id
            or observed.get("environment") != self.environment
            or not isinstance(observed.get("owner_token"), str)
            or not secrets.compare_digest(observed["owner_token"], self._owner_token)
        ):
            raise SamplingMutexOwnershipError(
                f"sampling marker at {self.path} is not owned by this lease; "
                "refusing automatic removal"
            )
        self.path.unlink()
        self._marker = None

    def __enter__(self) -> "SamplingMutex":
        return self.acquire()

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.release()


def _run(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _command(
    runner: CommandRunner, command: Sequence[str]
) -> subprocess.CompletedProcess[str]:
    result = runner(command)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}")
    return result


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    """Hash a JSON value with one stable, locale-independent encoding."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stable_id_record(source: str | None, token: str | None) -> dict[str, Any]:
    return {
        "source": source,
        "sha256": None if not token else _sha256_text(token),
        "raw_identifier_recorded": False,
    }


def _validate_host_binding(profile_id: str, profile: Mapping[str, Any]) -> None:
    binding = profile.get("host_binding")
    if not isinstance(binding, dict):
        raise ValueError(f"machine profile {profile_id!r} lacks host_binding")
    status = binding.get("status")
    if status not in {"registered", "unregistered"}:
        raise ValueError(
            f"machine profile {profile_id!r} has invalid host_binding.status"
        )
    engineering_policy = binding.get("unregistered_engineering_policy")
    if engineering_policy not in {"deny", "allow-explicitly-unbound"}:
        raise ValueError(
            f"machine profile {profile_id!r} has invalid unregistered "
            "engineering policy"
        )

    stable_id = binding.get("stable_machine_id")
    required_facts = binding.get("required_host_facts")
    if status == "registered":
        if not isinstance(stable_id, dict):
            raise ValueError(
                f"registered machine profile {profile_id!r} lacks stable_machine_id"
            )
        source = stable_id.get("source")
        digest = stable_id.get("sha256")
        if not isinstance(source, str) or not source:
            raise ValueError(
                f"registered machine profile {profile_id!r} has no identity source"
            )
        expected_sources = {
            "windows": "windows-machine-guid-sha256",
            "macos": "macos-platform-uuid-sha256",
            "wsl2": "linux-machine-id-sha256",
            "native-linux": "linux-machine-id-sha256",
        }
        if source != expected_sources.get(profile.get("environment")):
            raise ValueError(
                f"registered machine profile {profile_id!r} uses the wrong "
                "identity source for its environment"
            )
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(
                f"registered machine profile {profile_id!r} has invalid identity hash"
            )
        if not isinstance(required_facts, dict) or not required_facts:
            raise ValueError(
                f"registered machine profile {profile_id!r} lacks host facts"
            )
        for key, expected in required_facts.items():
            if not isinstance(key, str) or not key or not isinstance(
                expected, (str, int, bool)
            ):
                raise ValueError(
                    f"registered machine profile {profile_id!r} has invalid host fact"
                )
        mandatory_facts = {
            "system",
            "architecture",
            "logical_cpu_count",
            "cpu_model",
        }
        if profile.get("environment") == "windows":
            mandatory_facts.update(
                {
                    "os_release",
                    "os_version",
                    "processor_identifier",
                    "bios_version",
                    "system_manufacturer",
                    "system_product",
                    "baseboard_manufacturer",
                    "baseboard_product",
                    "installed_memory_bytes",
                }
            )
        elif profile.get("environment") == "macos":
            mandatory_facts.update(
                {"model", "memory_bytes", "os_product_version", "os_build"}
            )
        elif profile.get("environment") in {"wsl2", "native-linux"}:
            mandatory_facts.update({"memory_bytes", "os_release", "os_version"})
        missing = sorted(mandatory_facts.difference(required_facts))
        if missing:
            raise ValueError(
                f"registered machine profile {profile_id!r} lacks mandatory "
                f"host facts: {', '.join(missing)}"
            )
        if not isinstance(profile.get("baseline"), dict) or not profile["baseline"]:
            raise ValueError(
                f"registered machine profile {profile_id!r} lacks baseline facts"
            )
        if not isinstance(profile.get("toolchain"), dict) or not profile["toolchain"]:
            raise ValueError(
                f"registered machine profile {profile_id!r} lacks toolchain facts"
            )
    else:
        if stable_id not in (None, {}):
            raise ValueError(
                f"unregistered machine profile {profile_id!r} must not carry a "
                "trusted stable-machine identity"
            )
        if required_facts not in (None, {}):
            raise ValueError(
                f"unregistered machine profile {profile_id!r} must not carry "
                "trusted host facts"
            )


def _profile_contract_record(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return the registered facts whose change invalidates host evidence."""
    return {
        "profile_id": profile.get("id"),
        "physical_host_id": profile.get("physical_host_id"),
        "environment": profile.get("environment"),
        "provisioned": profile.get("provisioned"),
        "independent_hardware": profile.get("independent_hardware"),
        "roles": profile.get("roles"),
        "formal_claims": profile.get("formal_claims"),
        "host_binding": profile.get("host_binding"),
        "baseline": profile.get("baseline"),
        "filesystem": profile.get("filesystem"),
        "power": profile.get("power"),
        "scheduler": profile.get("scheduler"),
        "sampling_mutex": profile.get("sampling_mutex"),
        "toolchain": profile.get("toolchain"),
    }


def physical_host_identity_record(
    profile: Mapping[str, Any], probe: Mapping[str, Any]
) -> dict[str, Any]:
    """Compose a certificate-safe binding to one registered physical host.

    ``physical_host_id`` is HYLE's non-secret registry alias (for example
    ``W1``).  Platform identifiers such as MachineGuid or IOPlatformUUID are
    never copied; only their one-way digest and source class are retained.
    """
    raw_binding = profile.get("host_binding", {})
    binding = raw_binding if isinstance(raw_binding, Mapping) else {}
    raw_host = probe.get("host", {})
    host = raw_host if isinstance(raw_host, Mapping) else {}
    raw_observed_id = host.get("stable_machine_id", {})
    observed_id = raw_observed_id if isinstance(raw_observed_id, Mapping) else {}
    raw_expected_id = binding.get("stable_machine_id", {})
    expected_id = raw_expected_id if isinstance(raw_expected_id, Mapping) else {}
    raw_observed_facts = host.get("identity_facts", {})
    observed_facts = (
        raw_observed_facts if isinstance(raw_observed_facts, Mapping) else {}
    )
    raw_required_facts = binding.get("required_host_facts", {})
    required_facts = (
        raw_required_facts if isinstance(raw_required_facts, Mapping) else {}
    )
    registered = binding.get("status") == "registered"
    stable_match = bool(
        registered
        and observed_id.get("source") == expected_id.get("source")
        and observed_id.get("sha256") == expected_id.get("sha256")
        and observed_id.get("raw_identifier_recorded") is False
    )
    fact_mismatches = [
        key
        for key, expected in required_facts.items()
        if observed_facts.get(key) != expected
    ]
    facts_match = bool(registered and not fact_mismatches)
    if not registered:
        state = "UNREGISTERED"
    elif stable_match and facts_match:
        state = "REGISTERED_MATCH"
    else:
        state = "REGISTERED_MISMATCH"
    physical_host_id = str(profile.get("physical_host_id", ""))
    return {
        "schema": HOST_IDENTITY_SCHEMA,
        "profile_id": profile.get("id"),
        "physical_host_id": physical_host_id,
        "physical_host_key_sha256": _sha256_text(
            f"hyle-perf-physical-host-v1:{physical_host_id}"
        ),
        "environment": profile.get("environment"),
        "independent_hardware": profile.get("independent_hardware") is True,
        "profile_provisioned": profile.get("provisioned") is True,
        "registration_state": state,
        "stable_machine_id_source": observed_id.get("source"),
        "stable_machine_id_sha256": observed_id.get("sha256"),
        "stable_machine_id_matches_registered": stable_match,
        "required_host_facts_match": facts_match,
        "mismatched_host_facts": sorted(fact_mismatches),
        "registered_profile_contract_sha256": _sha256_json(
            _profile_contract_record(profile)
        ),
        "raw_identifier_recorded": False,
    }


def detect_environment(
    *,
    system: str | None = None,
    release: str | None = None,
    osrelease_text: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Return windows, macos, wsl2, or native-linux."""
    observed_system = system or platform.system()
    observed_release = release if release is not None else platform.release()
    observed_environ = os.environ if environ is None else environ

    if observed_system == "Windows":
        return "windows"
    if observed_system == "Darwin":
        return "macos"
    if observed_system != "Linux":
        return "unsupported"

    marker = osrelease_text
    if marker is None:
        try:
            marker = pathlib.Path("/proc/sys/kernel/osrelease").read_text(
                encoding="utf-8", errors="replace"
            )
        except OSError:
            marker = ""
    combined = f"{observed_release}\n{marker}".lower()
    if (
        "microsoft" in combined
        or "wsl" in combined
        or observed_environ.get("WSL_INTEROP")
        or observed_environ.get("WSL_DISTRO_NAME")
    ):
        return "wsl2"
    return "native-linux"


def load_registry(path: pathlib.Path | str = DEFAULT_REGISTRY_PATH) -> dict[str, Any]:
    registry_path = pathlib.Path(path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema") != SCHEMA:
        raise ValueError(f"unsupported machine registry schema in {registry_path}")
    profiles = registry.get("machine_profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("machine registry has no machine_profiles")
    physical_mutexes: dict[str, tuple[str, str, str]] = {}
    for profile_id, profile in profiles.items():
        if not isinstance(profile, dict):
            raise ValueError(f"machine profile {profile_id!r} is not an object")
        if profile.get("id") != profile_id:
            raise ValueError(f"machine profile key/id mismatch: {profile_id!r}")
        for required in ("physical_host_id", "environment", "roles", "provisioned"):
            if required not in profile:
                raise ValueError(f"machine profile {profile_id!r} lacks {required}")
        if not isinstance(profile["physical_host_id"], str) or not profile[
            "physical_host_id"
        ]:
            raise ValueError(
                f"machine profile {profile_id!r} has invalid physical_host_id"
            )
        if profile["environment"] not in {
            "windows",
            "wsl2",
            "macos",
            "native-linux",
        }:
            raise ValueError(
                f"machine profile {profile_id!r} has invalid environment"
            )
        if not isinstance(profile["provisioned"], bool):
            raise ValueError(
                f"machine profile {profile_id!r} has non-boolean provisioned"
            )
        roles = profile["roles"]
        if (
            not isinstance(roles, list)
            or len(set(roles)) != len(roles)
            or any(role not in {"correctness", "engineering", "formal"} for role in roles)
        ):
            raise ValueError(f"machine profile {profile_id!r} has invalid roles")
        formal_claims = profile.get("formal_claims", [])
        if (
            not isinstance(formal_claims, list)
            or len(set(formal_claims)) != len(formal_claims)
            or any(not isinstance(claim, str) or not claim for claim in formal_claims)
        ):
            raise ValueError(
                f"machine profile {profile_id!r} has invalid formal_claims"
            )
        if formal_claims and "formal" not in roles:
            raise ValueError(
                f"machine profile {profile_id!r} declares claims without formal role"
            )
        if "formal" in roles and profile.get("independent_hardware") is not True:
            raise ValueError(
                f"formal machine profile {profile_id!r} is not independent hardware"
            )
        _validate_host_binding(profile_id, profile)
        mutex = profile.get("sampling_mutex")
        if not isinstance(mutex, dict):
            raise ValueError(f"machine profile {profile_id!r} lacks sampling_mutex")
        mutex_key = mutex.get("key")
        storage_id = mutex.get("shared_storage_id")
        mutex_paths = mutex.get("paths")
        environment = profile["environment"]
        if (
            not isinstance(mutex_key, str)
            or not mutex_key
            or not isinstance(storage_id, str)
            or not storage_id
            or not isinstance(mutex_paths, dict)
            or not isinstance(mutex_paths.get(environment), str)
            or not mutex_paths[environment]
        ):
            raise ValueError(f"machine profile {profile_id!r} has invalid sampling_mutex")
        physical_host_id = str(profile["physical_host_id"])
        identity = (mutex_key, storage_id, json.dumps(mutex_paths, sort_keys=True))
        prior = physical_mutexes.setdefault(physical_host_id, identity)
        if prior != identity:
            raise ValueError(
                f"profiles for physical host {physical_host_id!r} do not share "
                "one sampling mutex key/storage"
            )
    return registry


def get_machine_profile(
    registry: Mapping[str, Any], profile_id: str
) -> Mapping[str, Any]:
    profile = registry.get("machine_profiles", {}).get(profile_id)
    if not isinstance(profile, dict):
        raise KeyError(f"unknown machine profile: {profile_id}")
    return profile


def _windows_physical_core_workers(
    profile: Mapping[str, Any], probe: Mapping[str, Any], worker_count: int, *,
    separate_caller: bool = False,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    """Select one logical sibling from N same-class Windows physical cores.

    The registered single-thread CPU anchors processor group and efficiency
    class.  Within that class, the lowest logical sibling represents each
    physical core; the anchor itself represents its own core.  Sorting by the
    representative logical index makes the selection deterministic without
    turning one host's CPU numbers into a module contract.
    An external caller can reserve its whole physical core outside the workers.
    """
    if isinstance(worker_count, bool) or not isinstance(worker_count, int) or \
            worker_count < 1:
        raise ValueError("worker_count must be a positive integer")
    scheduler = profile.get("scheduler", {})
    caller_cpu = scheduler.get("single_thread_logical_cpu")
    cpu_sets = probe.get("scheduler", {}).get("cpu_sets", [])
    caller = [
        item for item in cpu_sets
        if item.get("group") == 0
        and item.get("logical_processor_index") == caller_cpu
    ]
    if len(caller) != 1:
        raise ValueError("registered Windows caller CPU is absent or ambiguous")
    anchor = caller[0]
    eligible = [
        item for item in cpu_sets
        if item.get("group") == anchor.get("group")
        and item.get("efficiency_class") == anchor.get("efficiency_class")
        and isinstance(item.get("core_index"), int)
        and isinstance(item.get("logical_processor_index"), int)
    ]
    by_core: dict[int, list[Mapping[str, Any]]] = {}
    for item in eligible:
        by_core.setdefault(int(item["core_index"]), []).append(item)
    representatives = []
    for core_index, siblings in by_core.items():
        if core_index == anchor.get("core_index"):
            if separate_caller:
                continue
            representative = anchor
        else:
            representative = min(
                siblings, key=lambda item: item["logical_processor_index"])
        representatives.append(representative)
    representatives.sort(key=lambda item: item["logical_processor_index"])
    if len(representatives) < worker_count:
        raise ValueError(
            "Windows host has fewer than the requested distinct same-class "
            "physical cores")
    selected = representatives[:worker_count]
    if not separate_caller and anchor not in selected:
        # The registered caller is the machine contract's stable control CPU.
        # Keep it in the selected set while retaining deterministic ordering.
        selected[-1] = anchor
        selected.sort(key=lambda item: item["logical_processor_index"])
    return anchor, selected


def _windows_cross_core_pair(
    profile: Mapping[str, Any], probe: Mapping[str, Any]
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Select the registered caller plus the next same-class physical core.

    Keep the original cross-core-pair ordering contract independent from the
    N-worker selector: existing evidence expects the registered caller first
    and the next higher logical CPU on a distinct core second.
    """
    scheduler = profile.get("scheduler", {})
    caller_cpu = scheduler.get("single_thread_logical_cpu")
    cpu_sets = probe.get("scheduler", {}).get("cpu_sets", [])
    caller = [
        item for item in cpu_sets
        if item.get("group") == 0
        and item.get("logical_processor_index") == caller_cpu
    ]
    if len(caller) != 1:
        raise ValueError("registered Windows caller CPU is absent or ambiguous")
    anchor = caller[0]
    candidates = [
        item for item in cpu_sets
        if item.get("group") == anchor.get("group")
        and item.get("efficiency_class") == anchor.get("efficiency_class")
        and item.get("core_index") != anchor.get("core_index")
        and isinstance(item.get("logical_processor_index"), int)
    ]
    if not candidates:
        raise ValueError("Windows host has no distinct same-class physical core")
    higher = [
        item for item in candidates
        if item["logical_processor_index"] > caller_cpu
    ]
    worker = min(
        higher or candidates, key=lambda item: item["logical_processor_index"])
    return anchor, worker


def expected_child_policy(
    profile: Mapping[str, Any], *,
    sampling_topology: str = "single-logical-cpu",
    sampling_worker_count: int | None = None,
    probe: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe controls the benchmark launcher must apply to each child."""
    environment = profile.get("environment")
    scheduler = profile.get("scheduler", {})
    if environment == "windows":
        logical_cpu = scheduler.get("single_thread_logical_cpu")
        policy = {
            "environment": environment,
            "affinity_mask": (
                None if not isinstance(logical_cpu, int) else hex(1 << logical_cpu)
            ),
            "logical_cpu": logical_cpu,
            "cpu_set_id": scheduler.get("single_thread_cpu_set_id"),
            "required_efficiency_class": scheduler.get(
                "required_efficiency_class"
            ),
            "smt_sibling_logical_cpu": scheduler.get(
                "smt_sibling_logical_cpu"
            ),
            "priority": scheduler.get("priority"),
            "applied_by": "benchmark-child-launcher",
        }
        if sampling_topology == "single-logical-cpu":
            return policy
        if sampling_topology not in {
                "cross-core-pair", "n-worker-physical-cores",
                "caller-plus-n-worker-physical-cores"}:
            raise ValueError(
                f"unsupported Windows sampling topology: {sampling_topology!r}"
            )
        if probe is None:
            raise ValueError(
                f"{sampling_topology} sampling requires a host probe")
        if sampling_topology in {
                "n-worker-physical-cores",
                "caller-plus-n-worker-physical-cores"}:
            separate_caller = (
                sampling_topology == "caller-plus-n-worker-physical-cores")
            if (isinstance(sampling_worker_count, bool) or
                    not isinstance(sampling_worker_count, int) or
                    sampling_worker_count < 2):
                raise ValueError(
                    f"{sampling_topology} sampling requires worker_count >= 2")
            anchor, workers = _windows_physical_core_workers(
                profile, probe, sampling_worker_count,
                separate_caller=separate_caller)
            reserved = sorted(
                [anchor, *workers] if separate_caller else workers,
                key=lambda item: item["logical_processor_index"])
            logical_cpus = [
                int(cpu["logical_processor_index"]) for cpu in reserved]
            worker_cpus = [
                int(worker["logical_processor_index"]) for worker in workers]
            pointer_bits = ctypes.sizeof(ctypes.c_size_t) * 8
            if any(cpu >= pointer_bits for cpu in logical_cpus):
                raise ValueError(
                    "selected Windows worker CPU cannot be represented by the "
                    "process affinity mask")
            affinity_mask = sum(1 << cpu for cpu in logical_cpus)
            policy.update({
                "affinity_mask": hex(affinity_mask),
                "sampling_topology": sampling_topology,
                "sampling_worker_count": sampling_worker_count,
                "processor_group": anchor.get("group"),
                "logical_cpus": logical_cpus,
                "worker_logical_cpus": worker_cpus,
                "cpu_set_ids": [cpu.get("id") for cpu in reserved],
                "worker_cpu_set_ids": [
                    worker.get("id") for worker in workers],
                "physical_core_indices": [
                    cpu.get("core_index") for cpu in reserved],
                "efficiency_class": anchor.get("efficiency_class"),
                "caller_logical_cpu": int(
                    anchor["logical_processor_index"]),
                "worker_singleton_masks": [
                    hex(1 << cpu) for cpu in worker_cpus],
                "distinct_physical_cores": True,
                "smt_siblings_excluded": True,
                "applied_by": (
                    "benchmark-parent-mask-plus-worker-singleton-masks"),
            })
            if separate_caller:
                policy["reserved_physical_core_count"] = len(reserved)
                policy["caller_separate_physical_core"] = True
            return policy
        caller, worker = _windows_cross_core_pair(profile, probe)
        caller_cpu = int(caller["logical_processor_index"])
        worker_cpu = int(worker["logical_processor_index"])
        policy.update({
            "affinity_mask": hex((1 << caller_cpu) | (1 << worker_cpu)),
            "sampling_topology": sampling_topology,
            "logical_cpus": [caller_cpu, worker_cpu],
            "cpu_set_ids": [caller.get("id"), worker.get("id")],
            "caller_logical_cpu": caller_cpu,
            "worker_logical_cpu": worker_cpu,
            "distinct_physical_cores": True,
        })
        return policy
    if sampling_topology != "single-logical-cpu":
        raise ValueError(
            f"sampling topology {sampling_topology!r} is unsupported on "
            f"{environment!r}"
        )
    if environment == "macos":
        return {
            "environment": environment,
            "hard_affinity_supported": False,
            "qos": scheduler.get("default_qos"),
            "applied_by": "benchmark-child",
        }
    if environment == "wsl2":
        return {
            "environment": environment,
            "guest_affinity_only": True,
            "physical_core_class_verifiable": False,
            "applied_by": "benchmark-child-launcher",
        }
    return {
        "environment": environment,
        "hard_physical_affinity_verifiable": scheduler.get(
            "hard_physical_affinity_verifiable"
        ),
        "applied_by": "benchmark-child-launcher",
    }


def sampling_mutex(
    profile: Mapping[str, Any], *, environment: str | None = None
) -> SamplingMutex:
    """Create, but do not yet acquire, the registered sampling mutex."""
    observed_environment = environment or detect_environment()
    expected_environment = profile.get("environment")
    if observed_environment != expected_environment:
        raise SamplingMutexError(
            f"sampling mutex environment mismatch: expected {expected_environment}, "
            f"observed {observed_environment}"
        )
    mutex = profile.get("sampling_mutex")
    if not isinstance(mutex, Mapping):
        raise SamplingMutexError("machine profile has no sampling_mutex")
    paths = mutex.get("paths")
    if not isinstance(paths, Mapping) or not isinstance(
        paths.get(observed_environment), str
    ):
        raise SamplingMutexError(
            f"sampling mutex has no path for {observed_environment}"
        )
    return SamplingMutex(
        path=pathlib.Path(paths[observed_environment]),
        mutex_key=str(mutex.get("key")),
        physical_host_id=str(profile.get("physical_host_id")),
        profile_id=str(profile.get("id")),
        environment=observed_environment,
    )


def _stable_machine_id(environment: str, runner: CommandRunner) -> dict[str, Any]:
    """Return only a one-way digest; never expose the platform token."""
    token: str | None = None
    source: str | None = None
    if environment == "windows":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Cryptography",
                access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
            ) as key:
                token = str(winreg.QueryValueEx(key, "MachineGuid")[0]).strip()
                source = "windows-machine-guid-sha256"
        except (ImportError, FileNotFoundError, OSError):
            pass
    elif environment in {"wsl2", "native-linux"}:
        for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                value = pathlib.Path(candidate).read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            except OSError:
                continue
            if value:
                token = value
                source = "linux-machine-id-sha256"
                break
    elif environment == "macos":
        result = runner(["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
        match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', result.stdout)
        if result.returncode == 0 and match:
            token = match.group(1)
            source = "macos-platform-uuid-sha256"
    return _stable_id_record(source, token)


def _windows_identity_facts() -> dict[str, Any]:
    """Read stable, non-secret Windows host facts used by the registry gate."""
    facts: dict[str, Any] = {
        "system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "processor_identifier": platform.processor(),
    }
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            access=winreg.KEY_READ,
        ) as key:
            for registry_name, fact_name in (
                ("ProcessorNameString", "cpu_model"),
                ("VendorIdentifier", "cpu_vendor"),
                ("Identifier", "processor_identifier"),
            ):
                value = str(winreg.QueryValueEx(key, registry_name)[0]).strip()
                if value:
                    facts[fact_name] = value
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\BIOS",
            access=winreg.KEY_READ,
        ) as key:
            bios = winreg.QueryValueEx(key, "BIOSVersion")[0]
            if isinstance(bios, (list, tuple)):
                bios = " | ".join(
                    str(item).strip() for item in bios if str(item).strip()
                )
            else:
                bios = str(bios).strip()
            if bios:
                facts["bios_version"] = bios
            for registry_name, fact_name in (
                ("SystemManufacturer", "system_manufacturer"),
                ("SystemProductName", "system_product"),
                ("BaseBoardManufacturer", "baseboard_manufacturer"),
                ("BaseBoardProduct", "baseboard_product"),
            ):
                value = str(winreg.QueryValueEx(key, registry_name)[0]).strip()
                if value:
                    facts[fact_name] = value
    except (ImportError, FileNotFoundError, OSError):
        # Missing facts make a registered binding fail closed during evaluation.
        pass
    try:
        installed_kib = ctypes.c_ulonglong()
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_memory = kernel32.GetPhysicallyInstalledSystemMemory
        get_memory.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
        get_memory.restype = ctypes.c_bool
        if get_memory(ctypes.byref(installed_kib)):
            facts["installed_memory_bytes"] = int(installed_kib.value) * 1024
    except (AttributeError, OSError):
        pass
    return facts


def _command_value(runner: CommandRunner, command: Sequence[str]) -> str | None:
    result = runner(command)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def _macos_identity_facts(runner: CommandRunner) -> dict[str, Any]:
    facts: dict[str, Any] = {
        "system": "Darwin",
        "architecture": platform.machine(),
    }
    commands = {
        "model": ["sysctl", "-n", "hw.model"],
        "cpu_model": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "logical_cpu_count": ["sysctl", "-n", "hw.logicalcpu"],
        "memory_bytes": ["sysctl", "-n", "hw.memsize"],
        "os_product_version": ["sw_vers", "-productVersion"],
        "os_build": ["sw_vers", "-buildVersion"],
    }
    for key, command in commands.items():
        value = _command_value(runner, command)
        if value is None:
            continue
        if key in {"logical_cpu_count", "memory_bytes"}:
            try:
                facts[key] = int(value)
            except ValueError:
                continue
        else:
            facts[key] = value
    return facts


def _linux_cpu_model(cpuinfo: str) -> str | None:
    """Return a descriptive CPU model, never a numeric processor index."""
    candidates: dict[str, str] = {}
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        if not separator or not value.strip():
            continue
        normalized = key.strip().lower()
        text = value.strip()
        if normalized in {"model name", "hardware"}:
            candidates.setdefault(normalized, text)
        elif normalized == "processor" and not text.isdecimal():
            # Some ARM systems use "processor" for a descriptive model, while
            # x86/WSL use it for the logical index ("processor : 0").
            candidates.setdefault(normalized, text)
    return next(
        (candidates[key] for key in ("model name", "hardware", "processor")
         if key in candidates),
        None,
    )


def _linux_identity_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {
        "system": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        cpuinfo = pathlib.Path("/proc/cpuinfo").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        cpuinfo = ""
    cpu_model = _linux_cpu_model(cpuinfo)
    if cpu_model is not None:
        facts["cpu_model"] = cpu_model
    try:
        meminfo = pathlib.Path("/proc/meminfo").read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        meminfo = ""
    match = re.search(r"(?m)^MemTotal:\s+(\d+)\s+kB\s*$", meminfo)
    if match:
        facts["memory_bytes"] = int(match.group(1)) * 1024
    return facts


def _identity_facts(environment: str, runner: CommandRunner) -> dict[str, Any]:
    if environment == "windows":
        return _windows_identity_facts()
    if environment == "macos":
        return _macos_identity_facts(runner)
    if environment in {"wsl2", "native-linux"}:
        return _linux_identity_facts()
    return {
        "system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }


def _windows_volume_info(root: str) -> dict[str, Any]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetDriveTypeW.argtypes = [ctypes.c_wchar_p]
    kernel32.GetDriveTypeW.restype = ctypes.c_uint
    kernel32.GetVolumeInformationW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_wchar_p,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_wchar_p,
        ctypes.c_uint32,
    ]
    kernel32.GetVolumeInformationW.restype = ctypes.c_int

    filesystem = ctypes.create_unicode_buffer(64)
    if not kernel32.GetVolumeInformationW(
        root, None, 0, None, None, None, filesystem, len(filesystem)
    ):
        raise OSError(ctypes.get_last_error(), f"GetVolumeInformationW failed for {root}")
    drive_type_value = int(kernel32.GetDriveTypeW(root))
    drive_types = {
        0: "unknown",
        1: "no-root",
        2: "removable",
        3: "fixed",
        4: "remote",
        5: "optical",
        6: "ramdisk",
    }
    return {
        "filesystem": filesystem.value.upper(),
        "drive_type": drive_types.get(drive_type_value, "unknown"),
    }


def _windows_filesystem_record(
    label: str,
    path: pathlib.Path,
    volume_probe: Callable[[str], dict[str, Any]] = _windows_volume_info,
) -> dict[str, Any]:
    resolved = path.resolve()
    root = resolved.anchor
    if not root:
        return {
            "label": label,
            "path": str(resolved),
            "filesystem": None,
            "drive_type": "unknown",
            "local_fixed": False,
            "qualifies_for_performance": False,
        }
    volume = volume_probe(root)
    local_fixed = volume.get("drive_type") == "fixed"
    filesystem = str(volume.get("filesystem", "")).upper()
    return {
        "label": label,
        "path": str(resolved),
        "root": root,
        "filesystem": filesystem,
        "drive_type": volume.get("drive_type"),
        "local_fixed": local_fixed,
        "qualifies_for_performance": local_fixed and filesystem == "NTFS",
    }


def _parse_findmnt(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    filesystems = payload.get("filesystems", [])
    if not filesystems:
        raise ValueError("findmnt returned no filesystem")
    record = filesystems[0]
    filesystem = str(record.get("fstype", "")).lower()
    return {
        "mountpoint": record.get("target"),
        "source": record.get("source"),
        "filesystem": filesystem,
        "options": record.get("options"),
        "local_fixed": filesystem not in _REMOTE_FILESYSTEMS,
    }


def _linux_filesystem_record(
    label: str, path: pathlib.Path, environment: str, runner: CommandRunner
) -> dict[str, Any]:
    resolved = path.resolve()
    result = _command(
        runner,
        ["findmnt", "-J", "-T", str(resolved), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
    )
    record = _parse_findmnt(result.stdout)
    wsl_native = not re.match(r"^/mnt/[a-z](?:/|$)", str(resolved), re.I)
    qualifies = bool(record["local_fixed"])
    if environment == "wsl2":
        qualifies = qualifies and record["filesystem"] in {"ext2", "ext3", "ext4"}
        qualifies = qualifies and wsl_native
    return {
        "label": label,
        "path": str(resolved),
        **record,
        "qualifies_for_performance": qualifies,
    }


def _parse_diskutil_plist(text: str) -> dict[str, Any]:
    payload = plistlib.loads(text.encode("utf-8"))
    if payload.get("Error"):
        raise ValueError(payload.get("ErrorMessage", "diskutil returned an error"))
    filesystem = str(
        payload.get("FilesystemType") or payload.get("FilesystemName") or ""
    ).upper()
    internal = bool(payload.get("Internal"))
    return {
        "filesystem": filesystem,
        "mountpoint": payload.get("MountPoint"),
        "internal": internal,
        "solid_state": payload.get("SolidState"),
        "local_fixed": internal,
    }


def _macos_filesystem_record(
    label: str, path: pathlib.Path, runner: CommandRunner
) -> dict[str, Any]:
    resolved = path.resolve()
    df = _command(runner, ["df", "-P", str(resolved)])
    lines = [line for line in df.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"df returned no mount for {resolved}")
    device = lines[-1].split()[0]
    diskutil = _command(runner, ["diskutil", "info", "-plist", device])
    record = _parse_diskutil_plist(diskutil.stdout)
    return {
        "label": label,
        "path": str(resolved),
        "device": device,
        **record,
        "qualifies_for_performance": (
            record["filesystem"] == "APFS" and record["local_fixed"]
        ),
    }


def _parse_windows_power_scheme(text: str) -> dict[str, Any]:
    match = _POWER_GUID_RE.search(text)
    name_match = re.search(r"\(([^()]*)\)\s*$", text.strip())
    return {
        "active_scheme_guid": None if match is None else match.group(1).lower(),
        "active_scheme_name": None if name_match is None else name_match.group(1),
    }


def _windows_power_status() -> dict[str, Any]:
    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ac_line_status", ctypes.c_ubyte),
            ("battery_flag", ctypes.c_ubyte),
            ("battery_life_percent", ctypes.c_ubyte),
            ("system_status_flag", ctypes.c_ubyte),
            ("battery_life_time", ctypes.c_uint32),
            ("battery_full_life_time", ctypes.c_uint32),
        ]

    status = SystemPowerStatus()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        raise OSError(ctypes.get_last_error(), "GetSystemPowerStatus failed")
    sources = {0: "battery", 1: "ac", 255: "unknown"}
    return {
        "source": sources.get(int(status.ac_line_status), "unknown"),
        "battery_present": not bool(status.battery_flag & 128),
    }


def _parse_pmset_source(text: str) -> str:
    match = re.search(r"Now drawing from '([^']+)'", text)
    if not match:
        return "unknown"
    return "ac" if match.group(1).lower() == "ac power" else "battery"


def _parse_pmset_custom(text: str, source: str) -> dict[str, int]:
    section_name = "AC Power" if source == "ac" else "Battery Power"
    current: str | None = None
    values: dict[str, int] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.endswith(":") and line[:-1] in {"AC Power", "Battery Power"}:
            current = line[:-1]
            continue
        if current != section_name:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_]+)\s+(-?\d+)", line)
        if match:
            values[match.group(1)] = int(match.group(2))
    return values


def _macos_qos() -> dict[str, Any]:
    try:
        process = ctypes.CDLL(None)
        function = process.qos_class_self
        function.argtypes = []
        function.restype = ctypes.c_uint32
        value = int(function())
    except (AttributeError, OSError):
        return {"supported": False, "process_class": None, "raw_value": None}
    return {
        "supported": True,
        "process_class": _QOS_NAMES.get(value, f"unknown-{value:#x}"),
        "raw_value": value,
    }


@contextlib.contextmanager
def applied_macos_qos(requested: str | None):
    """Apply and restore the QoS contract around a macOS measurement host."""
    if requested is None:
        yield {"applied": False, "requested": None}
        return
    values = {name: value for value, name in _QOS_NAMES.items()}
    if requested not in {"interactive", "utility"}:
        raise ValueError(f"unsupported macOS QoS contract: {requested!r}")
    if platform.system() != "Darwin":
        raise RuntimeError("macOS QoS can only be applied on Darwin")
    library = ctypes.CDLL(None, use_errno=True)
    try:
        setter = library.pthread_set_qos_class_self_np
    except AttributeError as error:
        raise RuntimeError("macOS pthread QoS API is unavailable") from error
    setter.argtypes = [ctypes.c_uint32, ctypes.c_int]
    setter.restype = ctypes.c_int
    previous = _macos_qos()
    if not previous.get("supported") or not isinstance(
            previous.get("raw_value"), int):
        raise RuntimeError("cannot read current macOS process QoS")
    requested_value = values[requested]
    if setter(requested_value, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot apply macOS process QoS")
    observed = _macos_qos()
    if observed.get("process_class") != requested:
        setter(int(previous["raw_value"]), 0)
        raise RuntimeError(f"macOS QoS application mismatch: {observed!r}")
    try:
        yield {
            "applied": True,
            "requested": requested,
            "previous": previous["process_class"],
            "observed": observed["process_class"],
        }
    finally:
        if setter(int(previous["raw_value"]), 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot restore macOS process QoS")
        restored = _macos_qos()
        if restored.get("raw_value") != previous["raw_value"]:
            raise RuntimeError("macOS process QoS restoration mismatch")


def _windows_cpu_sets() -> list[dict[str, int]]:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetSystemCpuSetInformation.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p,
        ctypes.c_ulong,
    ]
    kernel32.GetSystemCpuSetInformation.restype = ctypes.c_int
    required = ctypes.c_ulong()
    handle = kernel32.GetCurrentProcess()
    kernel32.GetSystemCpuSetInformation(None, 0, ctypes.byref(required), handle, 0)
    if required.value == 0:
        return []
    buffer = (ctypes.c_ubyte * required.value)()
    if not kernel32.GetSystemCpuSetInformation(
        buffer, required.value, ctypes.byref(required), handle, 0
    ):
        return []
    records: list[dict[str, int]] = []
    offset = 0
    while offset < required.value:
        address = ctypes.addressof(buffer) + offset
        size = ctypes.c_uint32.from_address(address).value
        info_type = ctypes.c_uint32.from_address(address + 4).value
        if size == 0:
            break
        if info_type == 0 and size >= 20:
            records.append(
                {
                    "id": ctypes.c_uint32.from_address(address + 8).value,
                    "group": ctypes.c_uint16.from_address(address + 12).value,
                    "logical_processor_index": ctypes.c_uint8.from_address(
                        address + 14
                    ).value,
                    "core_index": ctypes.c_uint8.from_address(address + 15).value,
                    "efficiency_class": ctypes.c_uint8.from_address(
                        address + 18
                    ).value,
                }
            )
        offset += size
    return records


def _probe_filesystems(
    environment: str,
    paths: Iterable[tuple[str, pathlib.Path]],
    runner: CommandRunner,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for label, path in paths:
        if environment == "windows":
            records.append(_windows_filesystem_record(label, path))
        elif environment in {"wsl2", "native-linux"}:
            records.append(_linux_filesystem_record(label, path, environment, runner))
        elif environment == "macos":
            records.append(_macos_filesystem_record(label, path, runner))
        else:
            records.append(
                {
                    "label": label,
                    "path": str(path.resolve()),
                    "qualifies_for_performance": False,
                }
            )
    return records


def probe_machine(
    *,
    paths: Iterable[tuple[str, pathlib.Path | str]] | None = None,
    expected_qos: str | None = None,
    runner: CommandRunner = _run,
) -> dict[str, Any]:
    environment = detect_environment()
    normalized_paths = [
        (label, pathlib.Path(path))
        for label, path in (paths or (("workspace", pathlib.Path.cwd()),))
    ]
    filesystems = _probe_filesystems(environment, normalized_paths, runner)

    power: dict[str, Any]
    scheduler: dict[str, Any]
    if environment == "windows":
        scheme_result = _command(runner, ["powercfg", "/GETACTIVESCHEME"])
        query_result = _command(
            runner, ["powercfg", "/QUERY", "SCHEME_CURRENT", "SUB_PROCESSOR"]
        )
        power = {
            **_parse_windows_power_scheme(scheme_result.stdout),
            **_windows_power_status(),
            "processor_settings_sha256": _sha256_text(query_result.stdout),
        }
        cpu_sets = _windows_cpu_sets()
        scheduler = {
            "cpu_sets": cpu_sets,
            "maximum_efficiency_class": max(
                (item["efficiency_class"] for item in cpu_sets), default=None
            ),
            "hard_affinity_supported": True,
        }
    elif environment == "macos":
        battery = _command(runner, ["pmset", "-g", "batt"])
        source = _parse_pmset_source(battery.stdout)
        custom = _command(runner, ["pmset", "-g", "custom"])
        settings = _parse_pmset_custom(custom.stdout, source)
        power = {
            "source": source,
            "low_power_mode": settings.get("lowpowermode"),
            "settings": settings,
        }
        qos = _macos_qos()
        qos["expected_class"] = expected_qos
        qos["matches_expected"] = (
            None
            if expected_qos is None
            else qos["supported"] and qos["process_class"] == expected_qos
        )
        scheduler = {"hard_affinity_supported": False, "qos": qos}
    else:
        affinity = (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        )
        power = {
            "source": "host-managed" if environment == "wsl2" else "unknown"
        }
        scheduler = {
            "process_affinity": affinity,
            "hard_physical_affinity_verifiable": environment == "native-linux",
            "guest_vcpu_only": environment == "wsl2",
        }

    return {
        "schema": PROBE_SCHEMA,
        "observed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": environment,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "node_sha256": _sha256_text(platform.node()),
            "stable_machine_id": _stable_machine_id(environment, runner),
            "identity_facts": _identity_facts(environment, runner),
        },
        "filesystem": {
            "paths": filesystems,
            "all_paths_qualified": bool(filesystems)
            and all(item["qualifies_for_performance"] for item in filesystems),
        },
        "power": power,
        "scheduler": scheduler,
        "formal_capability": {
            "windows": environment == "windows",
            "macos-qos": environment == "macos",
            "native-linux": environment == "native-linux",
            "wsl2": environment == "wsl2",
            "independent-hardware": environment not in {"wsl2", "unsupported"},
        },
    }


def evaluate_machine_profile(
    profile: Mapping[str, Any],
    probe: Mapping[str, Any],
    *,
    role: str,
    formal_claim: str | None = None,
    sampling_topology: str = "single-logical-cpu",
    sampling_worker_count: int | None = None,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    environment = str(probe.get("environment"))
    if not profile.get("provisioned"):
        errors.append("machine profile is not provisioned")
    if environment != profile.get("environment"):
        errors.append(
            f"environment mismatch: expected {profile.get('environment')}, observed {environment}"
        )
    if role not in profile.get("roles", []):
        errors.append(f"role {role!r} is not allowed by the machine profile")

    identity = physical_host_identity_record(profile, probe)
    binding = profile.get("host_binding", {})
    if role in {"engineering", "formal"}:
        if binding.get("status") == "registered":
            if identity["registration_state"] != "REGISTERED_MATCH":
                if not identity["stable_machine_id_matches_registered"]:
                    errors.append(
                        "stable machine identity does not match the registered host"
                    )
                if not identity["required_host_facts_match"]:
                    mismatches = ", ".join(identity["mismatched_host_facts"])
                    errors.append(
                        "host facts do not match the registered baseline: " + mismatches
                    )
        elif role == "formal":
            errors.append(
                "formal evidence requires a registered stable-machine identity"
            )
        elif binding.get("unregistered_engineering_policy") == (
            "allow-explicitly-unbound"
        ):
            warnings.append(
                "engineering evidence is explicitly unbound from a registered "
                "stable-machine identity and cannot be promoted to Formal"
            )
        else:
            errors.append(
                "engineering evidence requires a registered stable-machine identity"
            )

    if role in {"engineering", "formal"}:
        filesystem = probe.get("filesystem", {})
        if not filesystem.get("all_paths_qualified"):
            errors.append("one or more paths are not on the required local filesystem")
        path_records = filesystem.get("paths", [])
        if not path_records:
            errors.append("performance roles require at least one probed path")
        filesystem_contract = profile.get("filesystem", {})
        required_filesystem = str(filesystem_contract.get("required", "")).upper()
        if required_filesystem and any(
            str(item.get("filesystem", "")).upper() != required_filesystem
            for item in path_records
        ):
            errors.append(f"filesystem does not match required {required_filesystem}")
        if filesystem_contract.get("require_local_fixed") and any(
            not item.get("local_fixed") for item in path_records
        ):
            errors.append("machine profile requires local fixed storage")
        forbidden_prefix = filesystem_contract.get("forbidden_prefix")
        if forbidden_prefix and any(
            pathlib.PurePosixPath(str(item.get("path", "/"))).is_relative_to(
                pathlib.PurePosixPath(forbidden_prefix)
            )
            for item in path_records
        ):
            errors.append(f"performance path is under forbidden prefix {forbidden_prefix}")

    if role == "formal":
        if profile.get("independent_hardware") is not True:
            errors.append("formal evidence requires independent registered hardware")
        if environment == "wsl2":
            errors.append(
                "WSL2 cannot issue a native-Linux or independent-hardware formal certificate"
            )
        claims = profile.get("formal_claims", [])
        if formal_claim is None:
            errors.append("formal role requires an explicit claim")
        elif formal_claim not in claims:
            errors.append(f"formal claim {formal_claim!r} is not allowed")
        elif not probe.get("formal_capability", {}).get(formal_claim):
            errors.append(
                f"probe cannot establish formal capability {formal_claim!r}"
            )

    power_contract = profile.get("power", {})
    observed_power = probe.get("power", {})
    if role in {"engineering", "formal"} and power_contract.get("require_ac"):
        if observed_power.get("source") != "ac":
            errors.append("AC power is required")
    expected_scheme = power_contract.get("active_scheme_guid")
    if role in {"engineering", "formal"} and expected_scheme:
        if observed_power.get("active_scheme_guid") != expected_scheme.lower():
            errors.append("active Windows power scheme does not match the profile")
    if role in {"engineering", "formal"} and power_contract.get(
        "require_low_power_mode_off"
    ):
        if observed_power.get("low_power_mode") != 0:
            errors.append("macOS low power mode must be off")

    scheduler_contract = profile.get("scheduler", {})
    if role in {"engineering", "formal"} and environment == "windows":
        logical_cpu = scheduler_contract.get("single_thread_logical_cpu")
        cpu_sets = probe.get("scheduler", {}).get("cpu_sets", [])
        selected = [
            item
            for item in cpu_sets
            if item.get("group") == 0
            and item.get("logical_processor_index") == logical_cpu
        ]
        maximum = probe.get("scheduler", {}).get("maximum_efficiency_class")
        if not selected:
            errors.append("registered Windows logical CPU is absent")
        elif scheduler_contract.get("required_efficiency_class") == "maximum" and any(
            item.get("efficiency_class") != maximum for item in selected
        ):
            errors.append("registered Windows logical CPU is not maximum efficiency class")
        expected_cpu_set_id = scheduler_contract.get("single_thread_cpu_set_id")
        if expected_cpu_set_id is not None and selected and any(
            item.get("id") != expected_cpu_set_id for item in selected
        ):
            errors.append("registered Windows CPU-set identity changed")
        if sampling_topology == "cross-core-pair":
            try:
                caller, worker = _windows_cross_core_pair(profile, probe)
            except ValueError as error:
                errors.append(str(error))
            else:
                if caller.get("core_index") == worker.get("core_index"):
                    errors.append("Windows cross-core pair shares one physical core")
                if (scheduler_contract.get("required_efficiency_class") ==
                        "maximum" and worker.get("efficiency_class") != maximum):
                    errors.append(
                        "Windows cross-core worker is not maximum efficiency class"
                    )
        elif sampling_topology in {
                "n-worker-physical-cores",
                "caller-plus-n-worker-physical-cores"}:
            separate_caller = (
                sampling_topology == "caller-plus-n-worker-physical-cores")
            try:
                caller, workers = _windows_physical_core_workers(
                    profile, probe, sampling_worker_count,
                    separate_caller=separate_caller)
            except (TypeError, ValueError) as error:
                errors.append(str(error))
            else:
                core_indices = [worker.get("core_index") for worker in workers]
                if len(set(core_indices)) != len(workers):
                    errors.append("Windows N-worker selection repeats a physical core")
                if separate_caller and caller.get("core_index") in core_indices:
                    errors.append("Windows external caller shares a worker core")
                if any(worker.get("group") != caller.get("group")
                       for worker in workers):
                    errors.append("Windows N-worker selection crosses processor groups")
                if (scheduler_contract.get("required_efficiency_class") ==
                        "maximum" and any(
                            worker.get("efficiency_class") != maximum
                            for worker in workers)):
                    errors.append(
                        "Windows N-worker selection is not maximum efficiency class")
    if role in {"engineering", "formal"} and environment == "macos":
        expected_qos = scheduler_contract.get("default_qos")
        qos = probe.get("scheduler", {}).get("qos", {})
        if expected_qos and (
            not qos.get("supported") or qos.get("process_class") != expected_qos
        ):
            errors.append(f"macOS process QoS does not match {expected_qos!r}")

    try:
        child_policy = expected_child_policy(
            profile, sampling_topology=sampling_topology,
            sampling_worker_count=sampling_worker_count, probe=probe
        )
    except ValueError as error:
        errors.append(str(error))
        child_policy = {
            "environment": environment,
            "sampling_topology": sampling_topology,
            "unavailable": True,
        }

    return {
        "gate": "PASS" if not errors else "FAIL",
        "profile_id": profile.get("id"),
        "role": role,
        "formal_claim": formal_claim,
        "physical_host_identity": identity,
        "expected_child_policy": child_policy,
        "errors": errors,
        "warnings": warnings,
    }


def _parse_path(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path must be LABEL=PATH")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("path must be LABEL=PATH")
    return label, pathlib.Path(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="probe a registered HYLE performance machine")
    parser.add_argument("--registry", type=pathlib.Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--machine", required=True, help="machine profile ID")
    parser.add_argument(
        "--role", choices=("correctness", "engineering", "formal"), required=True
    )
    parser.add_argument("--formal-claim")
    parser.add_argument("--apple-qos", choices=("interactive", "utility"))
    parser.add_argument("--path", action="append", type=_parse_path, default=[])
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)

    registry = load_registry(args.registry)
    try:
        profile = get_machine_profile(registry, args.machine)
    except KeyError:
        parser.error(f"unknown machine profile: {args.machine}")
    paths = args.path or [("workspace", pathlib.Path.cwd())]
    with applied_macos_qos(args.apple_qos):
        probe = probe_machine(paths=paths, expected_qos=args.apple_qos)
        evaluation = evaluate_machine_profile(
            profile, probe, role=args.role, formal_claim=args.formal_claim
        )
    output = {"probe": probe, "evaluation": evaluation}
    print(json.dumps(output, indent=None if args.compact else 2, sort_keys=True))
    return 0 if evaluation["gate"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
