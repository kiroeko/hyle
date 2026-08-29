#!/usr/bin/env python3
"""Tests for the shared performance machine registry and probe."""

from __future__ import annotations

import copy
import json
import pathlib
import plistlib
import subprocess
import sys
import tempfile
from unittest import mock


REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from hyle_perf import machine  # noqa: E402


def fake_probe(environment: str) -> dict:
    filesystem = {
        "windows": "NTFS",
        "wsl2": "ext4",
        "native-linux": "ext4",
        "macos": "APFS",
    }[environment]
    return {
        "environment": environment,
        "host": {
            "stable_machine_id": {
                "source": f"{environment}-test-id-sha256",
                "sha256": "0" * 64,
                "raw_identifier_recorded": False,
            },
            "identity_facts": {},
        },
        "filesystem": {
            "all_paths_qualified": True,
            "paths": [
                {
                    "path": "/workspace",
                    "filesystem": filesystem,
                    "local_fixed": True,
                    "qualifies_for_performance": True,
                }
            ],
        },
        "power": {},
        "scheduler": {},
        "formal_capability": {
            "windows": environment == "windows",
            "macos-qos": environment == "macos",
            "native-linux": environment == "native-linux",
            "wsl2": environment == "wsl2",
            "independent-hardware": environment not in {"wsl2", "unsupported"},
        },
    }


def bind_probe_to_registered_profile(probe: dict, profile: dict) -> dict:
    binding = profile["host_binding"]
    probe["host"] = {
        "stable_machine_id": {
            **binding["stable_machine_id"],
            "raw_identifier_recorded": False,
        },
        "identity_facts": dict(binding["required_host_facts"]),
    }
    return probe


def main() -> None:
    if machine.detect_environment(system="Windows") != "windows":
        raise AssertionError("Windows environment was not detected")
    if machine.detect_environment(system="Darwin") != "macos":
        raise AssertionError("macOS environment was not detected")
    if machine.detect_environment(
        system="Linux", release="6.8.0", osrelease_text="6.8.0-generic", environ={}
    ) != "native-linux":
        raise AssertionError("native Linux was not detected")
    for release, environment in (
        ("6.6.87.2-microsoft-standard-WSL2", {}),
        ("6.8.0", {"WSL_DISTRO_NAME": "Ubuntu-24.04"}),
    ):
        if machine.detect_environment(
            system="Linux", release=release, osrelease_text=release, environ=environment
        ) != "wsl2":
            raise AssertionError("WSL2 was allowed to masquerade as native Linux")

    registry = machine.load_registry(REPO / "scripts" / "perf-machines.json")
    profiles = registry["machine_profiles"]

    parsed_wsl_model = machine._linux_cpu_model(
        "processor : 0\n"
        "vendor_id : GenuineIntel\n"
        "model name : 13th Gen Intel(R) Core(TM) i9-13900K\n")
    if parsed_wsl_model != "13th Gen Intel(R) Core(TM) i9-13900K":
        raise AssertionError(
            f"Linux CPU parser selected a logical index: {parsed_wsl_model!r}")
    if machine._linux_cpu_model("processor : 0\nprocessor : 1\n") is not None:
        raise AssertionError("Linux CPU parser accepted a numeric processor index")

    qos_setter = mock.Mock(return_value=0)
    qos_library = mock.Mock()
    qos_library.pthread_set_qos_class_self_np = qos_setter
    with mock.patch.object(machine.platform, "system", return_value="Darwin"), \
            mock.patch.object(machine.ctypes, "CDLL", return_value=qos_library), \
            mock.patch.object(machine, "_macos_qos", side_effect=[
                {"supported": True, "process_class": "default",
                 "raw_value": 0x15},
                {"supported": True, "process_class": "interactive",
                 "raw_value": 0x21},
                {"supported": True, "process_class": "default",
                 "raw_value": 0x15},
            ]):
        with machine.applied_macos_qos("interactive") as applied:
            if applied["observed"] != "interactive":
                raise AssertionError("macOS QoS launcher did not attest its class")
    if qos_setter.call_args_list != [mock.call(0x21, 0), mock.call(0x15, 0)]:
        raise AssertionError(
            f"macOS QoS launcher did not apply/restore: {qos_setter.call_args_list}")
    if set(profiles) != {"W1-WIN", "W1-WSL", "M1-MAC", "L1-LINUX"}:
        raise AssertionError(f"machine registry IDs changed: {sorted(profiles)}")
    if profiles["W1-WSL"]["physical_host_id"] != profiles["W1-WIN"][
        "physical_host_id"
    ]:
        raise AssertionError("Windows and WSL lost their shared physical-host identity")
    expected_w1_hash = (
        "b4bbf09e3db097cf433da14a4ff16bef366dec28d42b27f8b303524267af5e97"
    )
    if (
        profiles["W1-WIN"]["host_binding"]["status"] != "registered"
        or profiles["W1-WIN"]["host_binding"]["stable_machine_id"]["sha256"]
        != expected_w1_hash
    ):
        raise AssertionError("W1 Windows profile lost its registered stable identity")
    if profiles["M1-MAC"]["host_binding"]["status"] != "unregistered":
        raise AssertionError("M1 became Formal-capable without an observed identity")

    for mutation, expected_text in (
        (
            lambda value: value["machine_profiles"]["W1-WIN"]["host_binding"].update(
                {
                    "stable_machine_id": {
                        "source": "windows-machine-guid-sha256",
                        "sha256": "not-a-hash",
                    }
                }
            ),
            "invalid identity hash",
        ),
        (
            lambda value: value["machine_profiles"]["W1-WIN"]["host_binding"].pop(
                "required_host_facts"
            ),
            "lacks host facts",
        ),
        (
            lambda value: value["machine_profiles"]["W1-WIN"]["host_binding"][
                "stable_machine_id"
            ].update({"source": "linux-machine-id-sha256"}),
            "wrong identity source",
        ),
        (
            lambda value: value["machine_profiles"]["W1-WIN"].update(
                {"independent_hardware": False}
            ),
            "not independent hardware",
        ),
    ):
        malformed = copy.deepcopy(registry)
        mutation(malformed)
        with tempfile.TemporaryDirectory() as temporary:
            path = pathlib.Path(temporary) / "registry.json"
            path.write_text(json.dumps(malformed), encoding="utf-8")
            try:
                machine.load_registry(path)
            except ValueError as error:
                if expected_text not in str(error):
                    raise AssertionError(
                        f"wrong fail-closed registry diagnostic: {error}"
                    ) from error
            else:
                raise AssertionError("malformed host identity registry was accepted")
    windows_mutex = profiles["W1-WIN"]["sampling_mutex"]
    wsl_mutex = profiles["W1-WSL"]["sampling_mutex"]
    if windows_mutex != wsl_mutex or windows_mutex["key"] != "W1":
        raise AssertionError("Windows and WSL do not share one frozen mutex contract")
    if windows_mutex["paths"] != {
        "windows": r"C:\hyle-perf-locks\W1.sampling.lock",
        "wsl2": "/mnt/c/hyle-perf-locks/W1.sampling.lock",
    }:
        raise AssertionError(f"W1 sampling paths drifted: {windows_mutex['paths']}")
    if machine.get_machine_profile(registry, "W1-WIN")["id"] != "W1-WIN":
        raise AssertionError("named machine profile lookup changed")
    try:
        machine.get_machine_profile(registry, "missing")
    except KeyError:
        pass
    else:
        raise AssertionError("missing machine profile did not fail closed")
    child_policy = machine.expected_child_policy(profiles["W1-WIN"])
    if (
        child_policy["affinity_mask"] != "0x4"
        or child_policy["cpu_set_id"] != 258
        or child_policy["priority"] != "above-normal"
    ):
        raise AssertionError(f"Windows child launch policy drifted: {child_policy}")
    registered_mutex = machine.sampling_mutex(
        profiles["W1-WIN"], environment="windows"
    )
    if str(registered_mutex.path) != r"C:\hyle-perf-locks\W1.sampling.lock":
        raise AssertionError(f"registered W1 mutex path drifted: {registered_mutex.path}")
    try:
        machine.sampling_mutex(profiles["W1-WIN"], environment="wsl2")
    except machine.SamplingMutexError:
        pass
    else:
        raise AssertionError("Windows profile acquired its WSL mutex path")

    with tempfile.TemporaryDirectory() as temporary:
        marker = pathlib.Path(temporary) / "W1.sampling.lock"
        first = machine.SamplingMutex(
            path=marker,
            mutex_key="W1",
            physical_host_id="W1",
            profile_id="W1-WIN",
            environment="windows",
        )
        with first as lease:
            if not marker.is_file() or not lease.held:
                raise AssertionError("sampling mutex did not create a persistent marker")
            manifest = lease.manifest_record()
            if (
                manifest["mutex_key"] != "W1"
                or manifest["raw_owner_token_recorded"] is not False
                or "owner_token" in manifest
            ):
                raise AssertionError(f"sampling mutex manifest leaked ownership: {manifest}")
            second = machine.SamplingMutex(
                path=marker,
                mutex_key="W1",
                physical_host_id="W1",
                profile_id="W1-WSL",
                environment="wsl2",
            )
            try:
                second.acquire()
            except machine.SamplingMutexBusy as error:
                if error.existing.get("mutex_key") != "W1":
                    raise AssertionError(
                        f"busy marker diagnostic lost its key: {error.existing}"
                    )
            else:
                raise AssertionError("two samplers acquired the same atomic marker")
            if not marker.exists():
                raise AssertionError("failed acquisition removed the active marker")
        if marker.exists() or first.held:
            raise AssertionError("owned sampling marker survived normal context exit")

        marker.write_text("not-json", encoding="utf-8")
        stale = machine.SamplingMutex(
            path=marker,
            mutex_key="W1",
            physical_host_id="W1",
            profile_id="W1-WIN",
            environment="windows",
        )
        try:
            stale.acquire()
        except machine.SamplingMutexBusy as error:
            if error.existing.get("valid") is not False:
                raise AssertionError("corrupt stale marker was treated as known ownership")
        else:
            raise AssertionError("corrupt stale marker was auto-recovered")
        if not marker.exists():
            raise AssertionError("unknown stale marker was automatically deleted")
        marker.unlink()

        tampered = machine.SamplingMutex(
            path=marker,
            mutex_key="M1",
            physical_host_id="M1",
            profile_id="M1-MAC",
            environment="macos",
        ).acquire()
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["owner_token"] = "foreign-owner"
        marker.write_text(json.dumps(payload), encoding="utf-8")
        try:
            tampered.release()
        except machine.SamplingMutexOwnershipError:
            pass
        else:
            raise AssertionError("tampered sampling marker was automatically deleted")
        if not marker.exists():
            raise AssertionError("foreign sampling marker was removed")

    wsl = fake_probe("wsl2")
    formal_wsl = machine.evaluate_machine_profile(
        profiles["W1-WSL"], wsl, role="formal", formal_claim="native-linux"
    )
    if formal_wsl["gate"] != "FAIL" or not any(
        "cannot issue" in error for error in formal_wsl["errors"]
    ):
        raise AssertionError(f"WSL acquired a native-Linux certificate: {formal_wsl}")
    engineering_wsl = machine.evaluate_machine_profile(
        profiles["W1-WSL"], wsl, role="engineering"
    )
    if engineering_wsl["gate"] != "PASS":
        raise AssertionError(f"qualified WSL engineering role failed: {engineering_wsl}")
    wsl_identity = engineering_wsl["physical_host_identity"]
    windows_alias_record = machine.physical_host_identity_record(
        profiles["W1-WIN"], fake_probe("windows")
    )
    if (
        wsl_identity["physical_host_key_sha256"]
        != windows_alias_record["physical_host_key_sha256"]
    ):
        raise AssertionError("Windows and WSL no longer identify one physical host")
    if wsl_identity["registration_state"] != "UNREGISTERED" or not any(
        "explicitly unbound" in warning for warning in engineering_wsl["warnings"]
    ):
        raise AssertionError(
            f"WSL engineering identity policy was not explicit: {engineering_wsl}"
        )

    fixed_ntfs = machine._windows_filesystem_record(
        "workspace",
        REPO,
        volume_probe=lambda _root: {"filesystem": "NTFS", "drive_type": "fixed"},
    )
    if not fixed_ntfs["qualifies_for_performance"]:
        raise AssertionError(f"fixed NTFS was rejected: {fixed_ntfs}")
    for volume in (
        {"filesystem": "NTFS", "drive_type": "remote"},
        {"filesystem": "ReFS", "drive_type": "fixed"},
    ):
        record = machine._windows_filesystem_record(
            "workspace", REPO, volume_probe=lambda _root, value=volume: value
        )
        if record["qualifies_for_performance"]:
            raise AssertionError(f"non-local/non-NTFS volume qualified: {record}")

    diskutil_payload = plistlib.dumps(
        {
            "FilesystemType": "apfs",
            "Internal": True,
            "SolidState": True,
            "MountPoint": "/System/Volumes/Data",
        }
    ).decode("utf-8")
    diskutil = machine._parse_diskutil_plist(diskutil_payload)
    if diskutil["filesystem"] != "APFS" or not diskutil["local_fixed"]:
        raise AssertionError(f"local APFS was not recognized: {diskutil}")
    external_payload = plistlib.dumps(
        {"FilesystemType": "apfs", "Internal": False, "SolidState": True}
    ).decode("utf-8")
    if machine._parse_diskutil_plist(external_payload)["local_fixed"]:
        raise AssertionError("external APFS acquired local-fixed status")

    pmset_battery = "Now drawing from 'AC Power'\n -InternalBattery-0 100%; charged"
    if machine._parse_pmset_source(pmset_battery) != "ac":
        raise AssertionError("macOS AC power was not recognized")
    pmset_custom = """
Battery Power:
 lowpowermode         1
 sleep                1
AC Power:
 lowpowermode         0
 sleep                1
"""
    if machine._parse_pmset_custom(pmset_custom, "ac").get("lowpowermode") != 0:
        raise AssertionError("macOS AC low-power state was parsed from the wrong section")

    def mac_runner(command):
        if command[:2] == ["df", "-P"]:
            output = (
                "Filesystem 512-blocks Used Available Capacity Mounted on\n"
                "/dev/disk3s5 100 10 90 10% /System/Volumes/Data\n"
            )
        elif command[:3] == ["diskutil", "info", "-plist"]:
            output = diskutil_payload
        elif command == ["pmset", "-g", "batt"]:
            output = pmset_battery
        elif command == ["pmset", "-g", "custom"]:
            output = pmset_custom
        elif command[:3] == ["ioreg", "-rd1", "-c"]:
            output = '"IOPlatformUUID" = "private-platform-uuid"\n'
        elif command == ["sysctl", "-n", "hw.model"]:
            output = "Mac16,12\n"
        elif command == ["sysctl", "-n", "machdep.cpu.brand_string"]:
            output = "Apple M4\n"
        elif command == ["sysctl", "-n", "hw.logicalcpu"]:
            output = "10\n"
        elif command == ["sysctl", "-n", "hw.memsize"]:
            output = "17179869184\n"
        elif command == ["sw_vers", "-productVersion"]:
            output = "26.5.2\n"
        elif command == ["sw_vers", "-buildVersion"]:
            output = "25F84\n"
        else:
            return subprocess.CompletedProcess(command, 1, "", "unexpected command")
        return subprocess.CompletedProcess(command, 0, output, "")

    with mock.patch.object(machine, "detect_environment", return_value="macos"), mock.patch.object(
        machine,
        "_macos_qos",
        return_value={
            "supported": True,
            "process_class": "interactive",
            "raw_value": 0x21,
        },
    ):
        integrated_mac = machine.probe_machine(
            paths=[("workspace", REPO)],
            expected_qos="interactive",
            runner=mac_runner,
        )
    if (
        not integrated_mac["filesystem"]["all_paths_qualified"]
        or integrated_mac["power"]["source"] != "ac"
        or integrated_mac["power"]["low_power_mode"] != 0
        or not integrated_mac["scheduler"]["qos"]["matches_expected"]
    ):
        raise AssertionError(f"integrated macOS probe lost a host gate: {integrated_mac}")
    encoded_mac = json.dumps(integrated_mac, sort_keys=True)
    if "private-platform-uuid" in encoded_mac:
        raise AssertionError("plaintext macOS platform UUID leaked into the probe")

    windows_probe = bind_probe_to_registered_profile(
        fake_probe("windows"), profiles["W1-WIN"]
    )
    windows_probe.update(
        {
            "power": {
                "source": "ac",
                "active_scheme_guid": "e9a42b02-d5df-448d-aa00-03f14749eb61",
            },
            "scheduler": {
                "maximum_efficiency_class": 1,
                "cpu_sets": [
                    {
                        "group": 0,
                        "logical_processor_index": 0,
                        "core_index": 0,
                        "efficiency_class": 1,
                        "id": 256,
                    },
                    {
                        "group": 0,
                        "logical_processor_index": 1,
                        "core_index": 0,
                        "efficiency_class": 1,
                        "id": 257,
                    },
                    {
                        "group": 0,
                        "logical_processor_index": 2,
                        "core_index": 2,
                        "efficiency_class": 1,
                        "id": 258,
                    },
                    {
                        "group": 0,
                        "logical_processor_index": 3,
                        "core_index": 2,
                        "efficiency_class": 1,
                        "id": 259,
                    },
                    {
                        "group": 0,
                        "logical_processor_index": 4,
                        "core_index": 4,
                        "efficiency_class": 1,
                        "id": 260,
                    },
                    {
                        "group": 0,
                        "logical_processor_index": 5,
                        "core_index": 4,
                        "efficiency_class": 1,
                        "id": 261,
                    },
                    *[
                        {
                            "group": 0,
                            "logical_processor_index": logical_cpu,
                            "core_index": logical_cpu,
                            "efficiency_class": 1,
                            "id": 256 + logical_cpu,
                        }
                        for logical_cpu in range(6, 16, 2)
                    ],
                ],
            },
        }
    )
    windows_formal = machine.evaluate_machine_profile(
        profiles["W1-WIN"], windows_probe, role="formal", formal_claim="windows"
    )
    if windows_formal["gate"] != "PASS":
        raise AssertionError(f"valid Windows formal profile failed: {windows_formal}")
    windows_engineering = machine.evaluate_machine_profile(
        profiles["W1-WIN"], windows_probe, role="engineering"
    )
    if windows_engineering["gate"] != "PASS":
        raise AssertionError(
            f"valid Windows engineering profile failed: {windows_engineering}"
        )
    cross_core = machine.expected_child_policy(
        profiles["W1-WIN"], sampling_topology="cross-core-pair",
        probe=windows_probe)
    if (
        cross_core["affinity_mask"] != "0x14"
        or cross_core["logical_cpus"] != [2, 4]
        or cross_core["cpu_set_ids"] != [258, 260]
        or cross_core["distinct_physical_cores"] is not True
    ):
        raise AssertionError(
            f"Windows cross-core child policy drifted: {cross_core}"
        )
    cross_core_gate = machine.evaluate_machine_profile(
        profiles["W1-WIN"], windows_probe, role="engineering",
        sampling_topology="cross-core-pair")
    if cross_core_gate["gate"] != "PASS":
        raise AssertionError(
            f"valid Windows cross-core profile failed: {cross_core_gate}"
        )
    n_worker = machine.expected_child_policy(
        profiles["W1-WIN"], sampling_topology="n-worker-physical-cores",
        sampling_worker_count=8, probe=windows_probe)
    expected_worker_cpus = list(range(0, 16, 2))
    if (
        n_worker["affinity_mask"] != "0x5555"
        or n_worker["worker_logical_cpus"] != expected_worker_cpus
        or n_worker["sampling_worker_count"] != 8
        or n_worker["worker_singleton_masks"] != [
            hex(1 << cpu) for cpu in expected_worker_cpus]
        or n_worker["distinct_physical_cores"] is not True
        or n_worker["smt_siblings_excluded"] is not True
    ):
        raise AssertionError(
            f"Windows N-worker child policy drifted: {n_worker}"
        )
    n_worker_gate = machine.evaluate_machine_profile(
        profiles["W1-WIN"], windows_probe, role="engineering",
        sampling_topology="n-worker-physical-cores",
        sampling_worker_count=8)
    if n_worker_gate["gate"] != "PASS":
        raise AssertionError(
            f"valid Windows N-worker profile failed: {n_worker_gate}"
        )
    identity_record = windows_formal["physical_host_identity"]
    if (
        identity_record["physical_host_id"] != "W1"
        or identity_record["registration_state"] != "REGISTERED_MATCH"
        or identity_record["independent_hardware"] is not True
        or identity_record["profile_provisioned"] is not True
        or identity_record["stable_machine_id_sha256"] != expected_w1_hash
        or identity_record["raw_identifier_recorded"] is not False
    ):
        raise AssertionError(f"certificate-safe host binding is incomplete: {identity_record}")
    encoded_identity = json.dumps(identity_record, sort_keys=True)
    if "MachineGuid" in encoded_identity or "IOPlatformUUID" in encoded_identity:
        raise AssertionError("physical-host binding exposed a raw platform identifier")
    malicious_probe = copy.deepcopy(windows_probe)
    malicious_probe["host"]["stable_machine_id"]["raw_identifier"] = (
        "do-not-copy-this-raw-machine-guid"
    )
    malicious_record = machine.physical_host_identity_record(
        profiles["W1-WIN"], malicious_probe
    )
    if "do-not-copy" in json.dumps(malicious_record, sort_keys=True):
        raise AssertionError("certificate-safe helper copied an undeclared raw ID")
    changed_profile = copy.deepcopy(profiles["W1-WIN"])
    changed_profile["baseline"]["bios"] = "future-bios"
    changed_record = machine.physical_host_identity_record(
        changed_profile, windows_probe
    )
    if (
        changed_record["registered_profile_contract_sha256"]
        == identity_record["registered_profile_contract_sha256"]
    ):
        raise AssertionError("baseline changes did not invalidate the profile binding")

    wrong_identity = copy.deepcopy(windows_probe)
    wrong_identity["host"]["stable_machine_id"]["sha256"] = "f" * 64
    wrong_identity_gate = machine.evaluate_machine_profile(
        profiles["W1-WIN"],
        wrong_identity,
        role="formal",
        formal_claim="windows",
    )
    if wrong_identity_gate["gate"] != "FAIL" or not any(
        "stable machine identity" in error
        for error in wrong_identity_gate["errors"]
    ):
        raise AssertionError("a foreign Windows machine acquired W1 Formal identity")
    wrong_cpu = copy.deepcopy(windows_probe)
    wrong_cpu["host"]["identity_facts"]["cpu_model"] = "lookalike CPU"
    wrong_cpu_gate = machine.evaluate_machine_profile(
        profiles["W1-WIN"], wrong_cpu, role="engineering"
    )
    if wrong_cpu_gate["gate"] != "FAIL" or not any(
        "cpu_model" in error for error in wrong_cpu_gate["errors"]
    ):
        raise AssertionError("a matching ID with a drifted CPU baseline qualified")
    registered_cpu_set = next(
        item for item in windows_probe["scheduler"]["cpu_sets"]
        if item["logical_processor_index"] == 2)
    registered_cpu_set["efficiency_class"] = 0
    if machine.evaluate_machine_profile(
        profiles["W1-WIN"], windows_probe, role="engineering"
    )["gate"] != "FAIL":
        raise AssertionError("engineering profile accepted a non-performance CPU set")
    registered_cpu_set["efficiency_class"] = 1
    windows_probe["filesystem"]["all_paths_qualified"] = False
    if machine.evaluate_machine_profile(
        profiles["W1-WIN"], windows_probe, role="formal", formal_claim="windows"
    )["gate"] != "FAIL":
        raise AssertionError("non-NTFS Windows formal probe passed")

    mac_probe = fake_probe("macos")
    mac_probe.update(
        {
            "power": {"source": "ac", "low_power_mode": 0},
            "scheduler": {
                "qos": {
                    "supported": True,
                    "process_class": "interactive",
                    "matches_expected": True,
                }
            },
        }
    )
    mac_formal = machine.evaluate_machine_profile(
        profiles["M1-MAC"], mac_probe, role="formal", formal_claim="macos-qos"
    )
    if mac_formal["gate"] != "FAIL" or not any(
        "registered stable-machine identity" in error
        for error in mac_formal["errors"]
    ):
        raise AssertionError(f"unregistered M1 self-certified as Formal: {mac_formal}")
    mac_engineering = machine.evaluate_machine_profile(
        profiles["M1-MAC"], mac_probe, role="engineering"
    )
    if mac_engineering["gate"] != "PASS" or not any(
        "explicitly unbound" in warning for warning in mac_engineering["warnings"]
    ):
        raise AssertionError(
            f"M1 engineering identity policy was not explicit: {mac_engineering}"
        )
    mac_probe["scheduler"]["qos"]["process_class"] = "utility"
    if machine.evaluate_machine_profile(
        profiles["M1-MAC"], mac_probe, role="engineering"
    )["gate"] != "FAIL":
        raise AssertionError("wrong macOS QoS acquired engineering eligibility")
    if machine.evaluate_machine_profile(
        profiles["M1-MAC"], mac_probe, role="formal", formal_claim="macos-qos"
    )["gate"] != "FAIL":
        raise AssertionError("wrong macOS QoS acquired a formal certificate")

    secret = "example-plaintext-machine-guid"
    stable_id = machine._stable_id_record("windows-machine-guid-sha256", secret)
    encoded = json.dumps(stable_id, sort_keys=True)
    if secret in encoded or stable_id.get("raw_identifier_recorded") is not False:
        raise AssertionError("plaintext MachineGuid leaked into the probe record")
    if stable_id["sha256"] != machine._sha256_text(secret):
        raise AssertionError("stable machine identity was not hashed deterministically")


if __name__ == "__main__":
    main()
