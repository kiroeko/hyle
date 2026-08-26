#!/usr/bin/env python3
"""Build and sample the legacy container_intrusive diagnostic instrument.

The runner fails closed on missing rows, changed inputs, incomplete execution
orders, unqualified A/A or known-delta controls, and unverified lifetime-loop
object code. Windows additionally qualifies a second MSVC placement image. It
never configures or builds unrelated HYLE components. Its output is diagnostic
and cannot replace the current v1 contract and common performance runner.
"""

import argparse
import ctypes
import datetime
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET


RAW_PREFIX = "# HYLE_BENCH_PAIR_RAW "
CHILD_TELEMETRY_PREFIX = "# HYLE_BENCH_CHILD_TELEMETRY "
PRODUCT_TELEMETRY_PREFIX = "# HYLE_CONTAINER_INTRUSIVE_RUN "
INSTRUMENT_TELEMETRY_PREFIX = \
    "# HYLE_CONTAINER_INTRUSIVE_INSTRUMENT_RUN "
FORMAL_MARGIN_PCT = 3.0
NEAR_DELTA_MAX_MARGIN_MULTIPLE = 2.0
LARGE_DELTA_RATIO_MIN = 0.75
LARGE_DELTA_RATIO_MAX = 0.85
FORMAL_BOOST_HEAD = "df8eeb90846648464e3c8e378e312f0cec772e83"
PRODUCT_CONTRACTS = [
    ("intrusive auto-unlink FIFO callback batch drain N=4096",
     "HYLE IntrusiveList::consume_all",
     "boost::intrusive auto_unlink clear_and_dispose", True),
    ("intrusive auto-unlink refill + arbitrary cancel N=4096",
     "HYLE IntrusiveList::remove", "boost::intrusive::erase", True),
    ("intrusive auto-unlink read-only traversal N=4096",
     "HYLE IntrusiveList iteration", "boost::intrusive iteration", True),
    ("intrusive auto-unlink owner teardown N=4096",
     "HYLE IntrusiveList auto detach",
     "boost::intrusive auto_unlink detach", True),
    ("intrusive auto-unlink linked-node retirement N=4096",
     "HYLE IntrusiveList linked node retirement",
     "boost::intrusive auto_unlink node retirement", True),
    ("intrusive safe-link FIFO callback batch drain N=4096",
     "HYLE SafeIntrusiveList::consume_all",
     "boost::intrusive safe_link clear_and_dispose", True),
    ("intrusive safe-link refill + arbitrary cancel N=4096",
     "HYLE SafeIntrusiveList::remove",
     "boost::intrusive safe_link erase", True),
    ("intrusive safe-link read-only traversal N=4096",
     "HYLE SafeIntrusiveList iteration",
     "boost::intrusive safe_link iteration", True),
    ("intrusive safe-link owner teardown N=4096",
     "HYLE SafeIntrusiveList owner detach",
     "boost::intrusive safe_link owner detach", True),
]
PRODUCT_PAIRS = [(baseline, rival)
                 for _, baseline, rival, _ in PRODUCT_CONTRACTS]
ALLOCATION_STRATA = [
    ("candidate-allocation-first", False),
    ("peer-allocation-first", True),
]
AA_PAIR = ("instrument identical A", "instrument identical B")
DELTA_PAIRS = [
    ("instrument control delta=0", "instrument control near=67/64 work"),
    ("instrument control delta=0", "instrument control large=80/64 work"),
]
TIMER_FLOOR_PAIR = (
    "empty aggregate timer floor A", "empty aggregate timer floor B")
PHASE_FLOOR_PAIR = (
    "phase observer fixed-call floor A",
    "phase observer fixed-call floor B")
CONTROL_CONTRACTS = [
    ("instrument A/A aggregate N=4096", *AA_PAIR, True),
    ("instrument known-delta aggregate N=4096", *DELTA_PAIRS[0], True),
    ("instrument known-delta aggregate N=4096", *DELTA_PAIRS[1], True),
    ("instrument empty aggregate N=4096", *TIMER_FLOOR_PAIR, False),
    ("instrument phase-observer floor N=4096", *PHASE_FLOOR_PAIR, False),
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_command(command):
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def command_text(command, cwd=None, encoding="utf-8"):
    result = subprocess.run(
        command, cwd=cwd, capture_output=True, text=True,
        encoding=encoding, errors="replace")
    if result.returncode != 0:
        raise RuntimeError(
            f"command exited {result.returncode}: "
            f"{display_command(command)}\n{result.stdout}{result.stderr}")
    return (result.stdout + result.stderr).strip()


def run_and_capture(command, environment, output_path):
    print(f"running: {display_command(command)}", flush=True)
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    output = result.stdout + result.stderr
    output_path.write_text(output, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"command exited {result.returncode}; see {output_path}")
    return output


def git_record(directory):
    status = command_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        directory)
    return {
        "head": command_text(["git", "rev-parse", "HEAD"], directory),
        "tree": command_text(["git", "rev-parse", "HEAD^{tree}"], directory),
        "clean": not bool(status),
        "status_porcelain": status.splitlines(),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def collect_input_hashes(repo):
    patterns = [
        "CMakeLists.txt",
        "cmake/HyleComponents.cmake",
        "cmake/HyleDependencies.cmake",
        "cmake/HyleDependencyLock.json",
        "cmake/HyleModule.cmake",
        "cmake/HyleOptions.cmake",
        "bench/CMakeLists.txt",
        "bench/nanobench_impl.cc",
        "bench/common/**/*",
        "bench/container_intrusive/**/*",
        "include/hyle/core/**/*",
        "include/hyle/container/intrusive_list.h",
        "scripts/bench-stat.py",
        "scripts/run-container-intrusive-bench-evidence.py",
        "third_party/nanobench/nanobench.h",
    ]
    paths = set()
    for pattern in patterns:
        paths.update(path for path in repo.glob(pattern) if path.is_file())
    return {
        path.relative_to(repo).as_posix(): sha256(path)
        for path in sorted(paths)
    }


def cmake_cache_values(build):
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        raise RuntimeError(f"configured build has no CMakeCache.txt: {build}")
    values = {}
    for line in cache.read_text(
            encoding="utf-8", errors="replace").splitlines():
        if not line or line.startswith(("#", "//")) or "=" not in line:
            continue
        key_type, value = line.split("=", 1)
        key = key_type.split(":", 1)[0]
        if (key.startswith("HYLE_") or key.startswith("CMAKE_GENERATOR") or
                key in {"CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE"}):
            values[key] = value
    expected = {
        "HYLE_COMPONENTS": "container_intrusive",
        "HYLE_BUILD_BENCH": "ON",
        "HYLE_BUILD_TESTS": "OFF",
        "HYLE_TEST_INSTRUMENTATION": "OFF",
        "HYLE_SANITIZER": "off",
        "HYLE_TRACE": "OFF",
        "HYLE_STATIC": "ON",
        "HYLE_BENCH_BOOST": "ON",
    }
    mismatches = {
        key: {"expected": wanted, "observed": values.get(key)}
        for key, wanted in expected.items() if values.get(key) != wanted
    }
    if mismatches:
        raise RuntimeError(
            "container_intrusive benchmark CMake contract mismatch: " +
            json.dumps(mismatches, sort_keys=True))
    return {"path": str(cache), "sha256": sha256(cache), "values": values}


def validate_cached_peer_path(cmake_contract, boost_dir):
    cached = cmake_contract["values"].get("HYLE_BOOST_DIR")
    if cached is None:
        raise RuntimeError("configured build does not record HYLE_BOOST_DIR")
    cached_path = pathlib.Path(cached).resolve()
    if os.path.normcase(str(cached_path)) != os.path.normcase(str(boost_dir)):
        raise RuntimeError(
            "configured Boost peer differs from --boost-dir: "
            f"cache={cached_path}, cli={boost_dir}")


def compiler_record(build):
    candidates = list(build.glob("CMakeFiles/*/CMakeCXXCompiler.cmake"))
    if not candidates:
        return None
    path = candidates[0]
    wanted = {
        "CMAKE_CXX_COMPILER", "CMAKE_CXX_COMPILER_ID",
        "CMAKE_CXX_COMPILER_VERSION",
        "CMAKE_CXX_COMPILER_ARCHITECTURE_ID", "CMAKE_CXX_SIMULATE_ID",
        "CMAKE_CXX_SIMULATE_VERSION",
    }
    values = {}
    for line in path.read_text(
            encoding="utf-8", errors="replace").splitlines():
        match = re.fullmatch(r'set\(([^ ]+) "(.*)"\)', line)
        if match and match.group(1) in wanted:
            values[match.group(1)] = match.group(2)
    compiler_path = pathlib.Path(values.get("CMAKE_CXX_COMPILER", ""))
    if not compiler_path.is_file():
        discovered = shutil.which(str(compiler_path))
        compiler_path = pathlib.Path(discovered) if discovered else None
    executable = None
    if compiler_path is not None and compiler_path.is_file():
        compiler_path = compiler_path.resolve()
        executable = {
            "path": str(compiler_path), "sha256": sha256(compiler_path)}
    return {
        "path": str(path), "sha256": sha256(path), "values": values,
        "executable": executable,
    }


def validate_formal_compiler(host_system, record, build_environment):
    if record is None or record.get("executable") is None:
        raise RuntimeError("formal build has no compiler identity record")
    values = record["values"]
    compiler_id = values.get("CMAKE_CXX_COMPILER_ID")
    version = values.get("CMAKE_CXX_COMPILER_VERSION", "")
    expected = {
        "Windows": ("MSVC", "19.50"),
        "Linux": ("GNU", "14."),
        "Darwin": ("AppleClang", "21."),
    }[host_system]
    if compiler_id != expected[0] or not version.startswith(expected[1]):
        raise RuntimeError(
            f"formal {host_system} compiler must be {expected[0]} "
            f"{expected[1]}x; observed {compiler_id} {version}")
    selected_cxx = build_environment.get("CXX")
    if host_system == "Linux" and selected_cxx:
        selected_path = shutil.which(selected_cxx)
        if selected_path is None or pathlib.Path(selected_path).resolve() != \
                pathlib.Path(record["executable"]["path"]).resolve():
            raise RuntimeError(
                "formal Linux CXX selector differs from the compiler bound "
                "by CMake")


def validate_formal_generator(host_system, cmake_contract):
    generator = cmake_contract["values"].get("CMAKE_GENERATOR")
    expected = {
        "Windows": "Visual Studio 18 2026",
        "Linux": "Ninja",
        "Darwin": "Ninja",
    }[host_system]
    if generator != expected:
        raise RuntimeError(
            f"formal {host_system} generator must be {expected!r}; "
            f"observed {generator!r}")


def _prior_log_strata_complete(logs, runs):
    expected_product_logs = {
        (shape, allocation, order)
        for shape in ("primary", "placement")
        for allocation, _ in ALLOCATION_STRATA
        for order in ("candidate-first", "peer-first")
    }
    product_logs = [
        entry for entry in logs if entry.get("kind") == "product"]
    instrument_logs = [
        entry for entry in logs if entry.get("kind") == "instrument"]
    observed_product_logs = {
        (entry.get("shape"), entry.get("allocation_order"),
         entry.get("execution_order"))
        for entry in product_logs
    }
    observed_instrument_logs = {
        entry.get("execution_order") for entry in instrument_logs
    }
    return (
        len(logs) == 10 and len(product_logs) == 8 and
        len(instrument_logs) == 2 and
        observed_product_logs == expected_product_logs and
        observed_instrument_logs == {"candidate-first", "peer-first"} and
        all(entry.get("raw_pair_records") == runs * len(PRODUCT_CONTRACTS)
            for entry in product_logs) and
        all(entry.get("raw_pair_records") == runs * len(CONTROL_CONTRACTS)
            for entry in instrument_logs))


def validate_prior_inconclusive_manifest(
        path, host_system, source_git, peer_git, input_hashes,
        host_identity, compiler, cmake_contract, build_environment,
        execution_controls, host_contract, apple_qos):
    manifest_path = pathlib.Path(path).resolve()
    if not manifest_path.is_file():
        raise RuntimeError(f"prior manifest is unavailable: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_qualification = "formal-clean-" + host_system.lower()
    if (manifest.get("record") !=
            "hyle_container_intrusive_bench_evidence_manifest" or
            manifest.get("schema") != 2 or
            manifest.get("qualification") != expected_qualification or
            manifest.get("host", {}).get("system") != host_system or
            manifest.get("runs_per_order") != 15 or
            not math.isclose(manifest.get("equivalence_margin_pct", -1.0),
                             FORMAL_MARGIN_PCT,
                             rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError("prior manifest is not the matching formal n=15 run")
    control_keys = {
        "Windows": {
            "system_affinity_mask", "applied_affinity_mask",
            "applied_priority_class", "selected_cpu_sets",
            "maximum_efficiency_class",
        },
        "Linux": {"applied_cpu_affinity", "applied_priority_class"},
        "Darwin": {"applied_priority_class"},
    }[host_system]
    prior_controls = manifest.get("execution_controls", {})
    if (manifest.get("host") != host_identity or
            manifest.get("compiler", {}).get("values") !=
            compiler.get("values") or
            manifest.get("compiler", {}).get("executable") !=
            compiler.get("executable") or
            manifest.get("build_environment") != build_environment or
            {key: prior_controls.get(key) for key in control_keys} !=
            {key: execution_controls.get(key) for key in control_keys} or
            manifest.get("host_contract_before_sampling") != host_contract or
            manifest.get("host_contract_stable_during_sampling") is not True or
            manifest.get("apple_qos_requested") != apple_qos or
            manifest.get("runtime_qos_telemetry_validated") is not True or
            manifest.get("runtime_strata_telemetry_validated") is not True):
        raise RuntimeError(
            "prior n=15 host/compiler/control population differs from the "
            "current n=31 experiment")
    contract_keys = {
        "CMAKE_GENERATOR", "CMAKE_GENERATOR_PLATFORM",
        "CMAKE_CXX_FLAGS", "CMAKE_CXX_FLAGS_RELEASE", "HYLE_BOOST_DIR",
        "HYLE_COMPONENTS", "HYLE_BUILD_BENCH", "HYLE_BUILD_TESTS",
        "HYLE_TEST_INSTRUMENTATION", "HYLE_SANITIZER", "HYLE_TRACE",
        "HYLE_STATIC", "HYLE_BENCH_BOOST",
    }
    prior_cmake = manifest.get("cmake_contract", {}).get("values", {})
    current_cmake = cmake_contract.get("values", {})
    if ({key: prior_cmake.get(key) for key in contract_keys} !=
            {key: current_cmake.get(key) for key in contract_keys}):
        raise RuntimeError(
            "prior n=15 CMake/compiler shape differs from current n=31")
    prior_source = manifest.get("source_git_at_start", {})
    prior_peer = manifest.get("peer_git_at_start", {})
    if (prior_source.get("head") != source_git["head"] or
            prior_source.get("tree") != source_git["tree"] or
            prior_peer.get("head") != peer_git["head"] or
            manifest.get("input_files_sha256") != input_hashes):
        raise RuntimeError(
            "prior n=15 manifest does not bind the current candidate/peer/inputs")
    if (prior_source.get("clean") is not True or
            manifest.get("source_git_at_end") != prior_source or
            prior_peer.get("clean") is not True or
            manifest.get("peer_git_at_end") != prior_peer or
            manifest.get("input_hashes_stable_during_run") is not True or
            manifest.get("audit_binding_stable") is not True or
            manifest.get("artifacts_stable_during_sampling") is not True or
            manifest.get("artifacts_stable_during_archive") is not True or
            manifest.get("audit_binding_before") !=
            manifest.get("artifacts_before_sampling") or
            manifest.get("artifacts_before_sampling") !=
            manifest.get("artifacts_after_sampling") or
            manifest.get("artifacts_after_sampling") !=
            manifest.get("artifacts_after_archive")):
        raise RuntimeError("prior n=15 provenance was not stable and clean")

    evidence_root = manifest_path.parent

    def require_evidence_file(record, label, path_key="path",
                              hash_key="sha256"):
        if not isinstance(record, dict):
            raise RuntimeError(f"prior n=15 {label} record is missing")
        candidate = pathlib.Path(record.get(path_key, "")).resolve()
        try:
            candidate.relative_to(evidence_root)
        except ValueError as error:
            raise RuntimeError(
                f"prior n=15 {label} escapes its evidence directory") \
                from error
        if (not candidate.is_file() or
                sha256(candidate) != record.get(hash_key)):
            raise RuntimeError(
                f"prior n=15 {label} is missing or changed")
        return candidate

    logs = manifest.get("logs", [])
    if not _prior_log_strata_complete(logs, 15):
        raise RuntimeError("prior n=15 raw log strata are incomplete")
    for index, entry in enumerate(logs):
        require_evidence_file(entry, f"raw log {index}")

    for key in (
            "no_interprocedural_optimization", "release_optimization",
            "artifact_architecture", "placement"):
        require_evidence_file(
            manifest.get(key), key, "record_path", "record_sha256")
    lifetime_records = manifest.get("lifetime_workload_codegen", {})
    if set(lifetime_records) != {"primary", "placement"}:
        raise RuntimeError("prior n=15 lifetime codegen records are incomplete")
    for label, record in lifetime_records.items():
        require_evidence_file(
            record, f"{label} lifetime codegen", "record_path",
            "record_sha256")
        require_evidence_file(
            record, f"{label} lifetime disassembly", "disassembly_path",
            "disassembly_file_sha256")

    archive = manifest.get("formal_artifact_archive", {})
    archive_root = pathlib.Path(archive.get("root", "")).resolve()
    try:
        archive_root.relative_to(evidence_root)
    except ValueError as error:
        raise RuntimeError("prior n=15 archive escapes evidence directory") \
            from error
    if archive.get("gate") != "PASS" or not archive_root.is_dir():
        raise RuntimeError("prior n=15 formal artifact archive is missing")
    archive_files = archive.get("files", [])
    for index, record in enumerate(archive_files):
        require_evidence_file(
            record, f"archive file {index}", "archived_path", "sha256")
    required_archive_labels = {
        "primary executable", "placement executable",
        "instrument executable", "primary_benchmark object",
        "placement_benchmark object", "instrument_benchmark object",
    }
    archive_by_label = {
        record.get("label"): record for record in archive_files
        if record.get("label") != "build metadata"
    }
    if set(archive_by_label) != required_archive_labels:
        raise RuntimeError("prior n=15 formal artifact archive is incomplete")
    frozen = manifest["artifacts_after_sampling"]
    expected_archive_hashes = {
        "primary executable": frozen["executables"]["primary"]["sha256"],
        "placement executable":
            frozen["executables"]["placement"]["sha256"],
        "instrument executable":
            frozen["executables"]["instrument"]["sha256"],
        "primary_benchmark object":
            frozen["objects"]["primary_benchmark"]["sha256"],
        "placement_benchmark object":
            frozen["objects"]["placement_benchmark"]["sha256"],
        "instrument_benchmark object":
            frozen["objects"]["instrument_benchmark"]["sha256"],
    }
    if any(archive_by_label[label].get("sha256") != expected_hash
           for label, expected_hash in expected_archive_hashes.items()):
        raise RuntimeError(
            "prior n=15 archive does not contain the sampled artifacts")
    archived_metadata_hashes = sorted(
        record.get("sha256") for record in archive_files
        if record.get("label") == "build metadata")
    if archived_metadata_hashes != sorted(
            frozen.get("build_binding_files_sha256", {}).values()):
        raise RuntimeError(
            "prior n=15 archive does not contain all bound build metadata")

    summary_record = manifest.get("summary", {})
    summary_path = require_evidence_file(summary_record, "summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("record") !=
            "hyle_container_intrusive_platform_summary" or
            summary.get("schema") != 3 or
            summary.get("platform") != host_system or
            summary.get("placement_evaluated") is not True or
            not math.isclose(summary.get("equivalence_margin_pct", -1.0),
                             FORMAL_MARGIN_PCT,
                             rel_tol=0.0, abs_tol=1e-12)):
        raise RuntimeError("prior n=15 summary schema/contract mismatch")
    if summary.get("platform_instrument_and_noninferiority_gate") != \
            "INCONCLUSIVE":
        raise RuntimeError("n=31 escalation requires an INCONCLUSIVE n=15 run")
    if summary.get("structural_workload_gate") != "PASS":
        raise RuntimeError("n=31 cannot override a structural-gate failure")
    controls = summary.get("controls", {})
    if controls.get("fixed_floor_gate") != "PASS":
        raise RuntimeError("n=31 cannot override a fixed-floor failure")

    margin = FORMAL_MARGIN_PCT / 100.0
    reciprocal_floor = 1.0 / (1.0 + margin)
    near_window_floor = 1.0 / (
        1.0 + NEAR_DELTA_MAX_MARGIN_MULTIPLE * margin)
    upper = 1.0 + FORMAL_MARGIN_PCT / 100.0
    aa = controls.get("aa", {})
    for order in ("candidate_first", "peer_first"):
        order_record = aa.get(order, {})
        median = order_record.get("ratio_median")
        if (order_record.get("exact_verdict") not in
                {"EQUIVALENT", "INCONCLUSIVE"} or
                not isinstance(median, (int, float)) or
                not math.isfinite(median) or
                not reciprocal_floor <= median <= upper):
            raise RuntimeError(
                "n=31 cannot override a directional or malformed A/A n=15")

    deltas = controls.get("known_delta", [])
    if len(deltas) != len(DELTA_PAIRS):
        raise RuntimeError("prior n=15 known-delta controls are incomplete")
    for order in ("candidate_first", "peer_first"):
        medians = []
        for index, delta in enumerate(deltas):
            order_record = delta.get(order, {})
            median = order_record.get("ratio_median")
            if (order_record.get("exact_verdict") not in
                    {"WIN", "INCONCLUSIVE"} or
                    not isinstance(median, (int, float)) or
                    not math.isfinite(median) or median >= reciprocal_floor):
                raise RuntimeError(
                    "n=31 cannot override an incorrectly directed or "
                    "sub-MDE known-delta n=15")
            if index == 0 and median < near_window_floor:
                raise RuntimeError(
                    "n=31 cannot turn an oversized near-MDE control into "
                    "instrument qualification")
            if index == 1 and not (
                    LARGE_DELTA_RATIO_MIN <= median <=
                    LARGE_DELTA_RATIO_MAX):
                raise RuntimeError(
                    "n=31 cannot override an out-of-window large known "
                    "delta n=15")
            medians.append(median)
        if not 0.0 < medians[1] < medians[0] < 1.0:
            raise RuntimeError(
                "n=31 cannot override a non-monotonic known-delta n=15")
    product_rows = summary.get("product_pairs", [])
    observed_pairs = {
        (row.get("baseline"), row.get("rival")) for row in product_rows
    }
    if (len(product_rows) != len(PRODUCT_PAIRS) or
            observed_pairs != set(PRODUCT_PAIRS)):
        raise RuntimeError("prior n=15 product contracts are incomplete")
    if any(row.get("platform_sota_noninferiority_gate") == "FAIL"
           for row in product_rows):
        raise RuntimeError("n=31 cannot override a product FAIL at n=15")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256(summary_path),
    }


def build_binding_hashes(build):
    paths = [build / "CMakeCache.txt", build / "build.ninja"]
    for target in (
            "hyle_container_intrusive_bench",
            "hyle_container_intrusive_bench_placement",
            "hyle_container_intrusive_instrument_bench"):
        paths.extend(build.rglob(target + ".vcxproj"))
        paths.extend(
            path for path in build.rglob("flags.make")
            if any(part.lower() == target.lower() + ".dir"
                   for part in path.parts))
    return {
        path.relative_to(build).as_posix(): sha256(path)
        for path in sorted(set(path for path in paths if path.is_file()))
    }


def llvm_objdump_identity(tool):
    if tool is None:
        return None
    version = command_text([str(tool), "--version"])
    if "llvm" not in version.lower():
        raise RuntimeError("the disassembly gate requires an LLVM objdump")
    version_lines = [line.strip() for line in version.splitlines()
                     if "version" in line.lower()]
    if not version_lines:
        raise RuntimeError("LLVM objdump did not report a version")
    return {
        "path": str(tool),
        "sha256": sha256(tool),
        "version": version_lines[0],
        "version_output_sha256": hashlib.sha256(
            version.encode("utf-8")).hexdigest(),
    }


def artifact_binding(primary, instrument, placement, primary_object,
                     instrument_object, placement_object, build, tool=None):
    def file_record(path):
        return None if path is None else {
            "path": str(path), "sha256": sha256(path)}
    return {
        "executables": {
            "primary": file_record(primary),
            "placement": file_record(placement),
            "instrument": file_record(instrument),
        },
        "objects": {
            "primary_benchmark": file_record(primary_object),
            "instrument_benchmark": file_record(instrument_object),
            "placement_benchmark": file_record(placement_object),
        },
        "llvm_objdump": llvm_objdump_identity(tool),
        "build_binding_files_sha256": build_binding_hashes(build),
    }


def archive_formal_artifacts(output_dir, artifact_record, build):
    archive = output_dir / "artifacts"
    archive.mkdir(parents=False, exist_ok=False)
    records = []

    def copy_bound(label, source, expected_hash, destination_name):
        if source is None:
            return
        source_path = pathlib.Path(source)
        destination = archive / destination_name
        shutil.copy2(source_path, destination)
        source_hash = sha256(source_path)
        destination_hash = sha256(destination)
        if (source_hash != expected_hash or
                destination_hash != expected_hash):
            raise RuntimeError(f"archived artifact changed while copying: {label}")
        records.append({
            "label": label, "source_path": str(source_path),
            "archived_path": str(destination),
            "sha256": destination_hash,
        })

    for label, bound in artifact_record["executables"].items():
        if bound is not None:
            suffix = pathlib.Path(bound["path"]).suffix
            copy_bound(label + " executable", bound["path"],
                       bound["sha256"],
                       label + "-executable" + suffix)
    for label, bound in artifact_record["objects"].items():
        if bound is not None:
            suffix = "".join(pathlib.Path(bound["path"]).suffixes)
            copy_bound(label + " object", bound["path"], bound["sha256"],
                       label + "-object" + suffix)

    metadata_root = archive / "build-metadata"
    for relative, expected_hash in artifact_record[
            "build_binding_files_sha256"].items():
        source = build / pathlib.PurePosixPath(relative)
        destination = metadata_root / pathlib.PurePosixPath(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        actual_hash = sha256(destination)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"archived build metadata changed while copying: {relative}")
        records.append({
            "label": "build metadata", "source_path": str(source),
            "archived_path": str(destination), "sha256": actual_hash,
        })
    return {"gate": "PASS", "root": str(archive), "files": records}


def find_executable(build, config, stem):
    suffix = ".exe" if os.name == "nt" else ""
    candidates = [
        build / "bin" / config / (stem + suffix),
        build / "bin" / (stem + suffix),
        build / config / (stem + suffix),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot find {stem + suffix} under {build}")


def find_benchmark_object(build, config, target, source_stem):
    # VS names the source object `<stem>.obj`; CMake's Unix Makefiles and
    # Ninja conventionally use `<stem>.cc.o` (with `<stem>.o` also possible).
    # Search the complete closed set and fail on stale/mixed-generator output.
    object_names = {
        source_stem + ".obj", source_stem + ".o", source_stem + ".cc.o"}
    discovered = set()
    for object_name in object_names:
        discovered.update(build.rglob(object_name))
    candidates = [
        path for path in discovered if path.is_file() and
        any(part.lower() == target.lower() + ".dir" for part in path.parts)
    ]
    configured = [
        path for path in candidates
        if config.lower() in [part.lower() for part in path.parts]
    ]
    candidates = configured or candidates
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one {target} object named "
            f"{sorted(object_names)!r}; "
            f"found {len(candidates)}")
    return candidates[0].resolve()


def windows_kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t)]
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetProcessAffinityMask.argtypes = [
        ctypes.c_void_p, ctypes.c_size_t]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.GetPriorityClass.restype = ctypes.c_uint32
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    kernel32.GetSystemCpuSetInformation.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p, ctypes.c_ulong]
    kernel32.GetSystemCpuSetInformation.restype = ctypes.c_int
    return kernel32


def windows_cpu_sets(kernel32):
    required = ctypes.c_ulong()
    handle = kernel32.GetCurrentProcess()
    kernel32.GetSystemCpuSetInformation(
        None, 0, ctypes.byref(required), handle, 0)
    if required.value == 0:
        return []
    buffer = (ctypes.c_ubyte * required.value)()
    if not kernel32.GetSystemCpuSetInformation(
            buffer, required.value, ctypes.byref(required), handle, 0):
        return []
    records = []
    offset = 0
    while offset < required.value:
        address = ctypes.addressof(buffer) + offset
        size = ctypes.c_uint32.from_address(address).value
        info_type = ctypes.c_uint32.from_address(address + 4).value
        if size == 0:
            break
        if info_type == 0 and size >= 20:
            records.append({
                "id": ctypes.c_uint32.from_address(address + 8).value,
                "group": ctypes.c_uint16.from_address(address + 12).value,
                "logical_processor_index":
                    ctypes.c_uint8.from_address(address + 14).value,
                "core_index": ctypes.c_uint8.from_address(address + 15).value,
                "last_level_cache_index":
                    ctypes.c_uint8.from_address(address + 16).value,
                "numa_node_index":
                    ctypes.c_uint8.from_address(address + 17).value,
                "efficiency_class":
                    ctypes.c_uint8.from_address(address + 18).value,
            })
        offset += size
    return records


def apply_execution_controls(affinity_text, priority):
    affinity = None if affinity_text is None else int(affinity_text, 0)
    if affinity is not None and affinity <= 0:
        raise ValueError("affinity mask must select at least one CPU")
    if os.name == "nt":
        kernel32 = windows_kernel32()
        handle = kernel32.GetCurrentProcess()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
                handle, ctypes.byref(process_mask),
                ctypes.byref(system_mask)):
            raise RuntimeError("GetProcessAffinityMask failed")
        if affinity is not None:
            if affinity & ~system_mask.value:
                raise ValueError(
                    f"affinity {hex(affinity)} exceeds system mask "
                    f"{hex(system_mask.value)}")
            if not kernel32.SetProcessAffinityMask(handle, affinity):
                raise RuntimeError("SetProcessAffinityMask failed")
        priority_values = {
            "normal": 0x00000020,
            "above-normal": 0x00008000,
            "high": 0x00000080,
        }
        if not kernel32.SetPriorityClass(handle, priority_values[priority]):
            raise RuntimeError("SetPriorityClass failed")
        cpu_sets = windows_cpu_sets(kernel32)
        if affinity is not None and any(
                record["group"] != 0 for record in cpu_sets):
            raise RuntimeError(
                "explicit affinity on a multi-processor-group Windows host "
                "is unsupported by this certificate runner")
        selected = []
        if affinity is not None:
            selected = [
                record for record in cpu_sets
                if record["group"] == 0 and
                affinity & (1 << record["logical_processor_index"])
            ]
        if affinity is not None and not selected:
            raise RuntimeError("selected affinity is absent from CPU-set data")
        max_efficiency = max(
            (record["efficiency_class"] for record in cpu_sets),
            default=None)
        if (selected and max_efficiency is not None and
                any(record["efficiency_class"] != max_efficiency
                    for record in selected)):
            raise RuntimeError(
                "selected affinity is not in the maximum efficiency class")
        return {
            "process_affinity_mask_before": hex(process_mask.value),
            "system_affinity_mask": hex(system_mask.value),
            "applied_affinity_mask": (
                None if affinity is None else hex(affinity)),
            "applied_priority_class": priority,
            "cpu_sets": cpu_sets,
            "selected_cpu_sets": selected,
            "maximum_efficiency_class": max_efficiency,
        }
    if affinity is not None:
        cpus = {
            bit for bit in range(affinity.bit_length())
            if affinity & (1 << bit)
        }
        os.sched_setaffinity(0, cpus)
    if priority != "normal":
        raise RuntimeError("non-Windows priority control is not implemented")
    return {
        "applied_cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None),
        "applied_priority_class": priority,
    }


def host_telemetry(host_system, affinity_text):
    def read_optional(path):
        candidate = pathlib.Path(path)
        return (candidate.read_text(encoding="utf-8", errors="replace").strip()
                if candidate.is_file() else None)

    record = {"uname": list(platform.uname())}
    if host_system == "Linux":
        selected_cpu = None
        if affinity_text is not None:
            affinity = int(affinity_text, 0)
            if affinity.bit_count() == 1:
                selected_cpu = affinity.bit_length() - 1
        record.update({
            "selected_logical_cpu": selected_cpu,
            "selected_cpu_scaling_governor": (
                None if selected_cpu is None else read_optional(
                    f"/sys/devices/system/cpu/cpu{selected_cpu}/cpufreq/"
                    "scaling_governor")),
            "selected_cpu_thread_siblings": (
                None if selected_cpu is None else read_optional(
                    f"/sys/devices/system/cpu/cpu{selected_cpu}/topology/"
                    "thread_siblings_list")),
            "smt_active": read_optional("/sys/devices/system/cpu/smt/active"),
            "numa_nodes_online": read_optional(
                "/sys/devices/system/node/online"),
            "cpuinfo_sha256": (
                sha256(pathlib.Path("/proc/cpuinfo"))
                if pathlib.Path("/proc/cpuinfo").is_file() else None),
        })
    return record


def build_environment_record(host_system, formal):
    keys = [
        "CC", "CXX", "CPPFLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS",
        "SDKROOT", "CMAKE_TOOLCHAIN_FILE", "CL", "_CL_", "LINK",
        "_LINK_", "LD_PRELOAD", "LD_LIBRARY_PATH",
        "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    ]
    values = {key: os.environ.get(key) for key in keys}
    if formal:
        allowed_nonempty = {"CC", "CXX"} if host_system == "Linux" else set()
        forbidden = {
            key: value for key, value in values.items()
            if value and key not in allowed_nonempty
        }
        if forbidden:
            raise RuntimeError(
                "formal build rejects compiler/linker/runtime environment "
                "overrides: " + ", ".join(sorted(forbidden)))
        if host_system == "Linux":
            expected = {"CC": "gcc-14", "CXX": "g++-14"}
            unexpected = {
                key: value for key, value in values.items()
                if key in expected and value and
                pathlib.Path(value).name != expected[key]
            }
            if unexpected:
                raise RuntimeError(
                    "formal Linux compiler selectors must be gcc-14/g++-14: "
                    + ", ".join(sorted(unexpected)))
    return values


def _optional_command_text(command):
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, encoding="utf-8", errors="replace", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _stable_machine_token(host_system):
    if host_system == "Windows":
        try:
            import winreg
            with winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Microsoft\Cryptography",
                    access=winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
                return "windows-machine-guid", winreg.QueryValueEx(
                    key, "MachineGuid")[0]
        except (ImportError, FileNotFoundError, OSError):
            return None, None
    if host_system == "Linux":
        for candidate in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            path = pathlib.Path(candidate)
            if path.is_file():
                value = path.read_text(
                    encoding="utf-8", errors="replace").strip()
                if value:
                    return candidate, value
        return None, None
    if host_system == "Darwin":
        text = _optional_command_text(
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"])
        match = (None if text is None else re.search(
            r'"IOPlatformUUID"\s*=\s*"([^"]+)"', text))
        return (("IOPlatformUUID", match.group(1))
                if match is not None else (None, None))
    return None, None


def _cpu_identity_record(host_system):
    if host_system == "Linux":
        path = pathlib.Path("/proc/cpuinfo")
        if not path.is_file():
            return None
        stable_fields = {
            "processor", "vendor_id", "cpu family", "model", "stepping",
            "microcode", "physical id", "core id", "cpu cores", "siblings",
            "model name",
        }
        processors = []
        current = {}
        for line in path.read_text(
                encoding="utf-8", errors="replace").splitlines() + [""]:
            if not line.strip():
                if current:
                    processors.append(current)
                    current = {}
                continue
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if key in stable_fields:
                current[key] = value
        encoded = json.dumps(
            processors, sort_keys=True, separators=(",", ":"))
        return {
            "schema": "linux-proc-cpuinfo-stable-fields-v1",
            "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "logical_processor_records": len(processors),
            "model_names": sorted({
                record["model name"] for record in processors
                if "model name" in record}),
            "microcodes": sorted({
                record["microcode"] for record in processors
                if "microcode" in record}),
        }
    if host_system == "Darwin":
        values = {
            key: _optional_command_text(["sysctl", "-n", key])
            for key in (
                "hw.model", "machdep.cpu.brand_string", "hw.physicalcpu",
                "hw.logicalcpu", "hw.packages")
        }
        return {key: value for key, value in values.items()
                if value is not None}
    if host_system == "Windows":
        return {
            "processor_identifier": os.environ.get(
                "PROCESSOR_IDENTIFIER", ""),
            "processor_architecture": os.environ.get(
                "PROCESSOR_ARCHITECTURE", ""),
        }
    return None


def host_identity_record(formal=False):
    host_system = platform.system()
    token_source, token = _stable_machine_token(host_system)
    node = platform.node()
    if formal and (not token_source or not token or not node):
        raise RuntimeError(
            "formal evidence requires a stable OS machine id and host name")
    return {
        "system": host_system,
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor()
        or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "logical_cpu_count": os.cpu_count(),
        "node_sha256": hashlib.sha256(node.encode("utf-8")).hexdigest(),
        "stable_machine_id_source": token_source,
        "stable_machine_id_sha256": (
            None if token is None else
            hashlib.sha256(str(token).encode("utf-8")).hexdigest()),
        "cpu_identity": _cpu_identity_record(host_system),
    }


def stable_host_contract_snapshot(host_system, affinity_text):
    def read_optional(path):
        candidate = pathlib.Path(path)
        return (candidate.read_text(
            encoding="utf-8", errors="replace").strip()
                if candidate.is_file() else None)

    if host_system == "Windows":
        kernel32 = windows_kernel32()
        handle = kernel32.GetCurrentProcess()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
                handle, ctypes.byref(process_mask),
                ctypes.byref(system_mask)):
            raise RuntimeError("GetProcessAffinityMask failed")
        return {
            "process_affinity_mask": hex(process_mask.value),
            "system_affinity_mask": hex(system_mask.value),
            "priority_class": kernel32.GetPriorityClass(handle),
            "active_power_scheme": command_text(
                ["powercfg", "/GETACTIVESCHEME"], encoding="utf-8"),
            "cpu_sets": windows_cpu_sets(kernel32),
        }
    if host_system == "Linux":
        selected_cpu = None
        if affinity_text is not None:
            affinity = int(affinity_text, 0)
            if affinity.bit_count() == 1:
                selected_cpu = affinity.bit_length() - 1
        return {
            "process_affinity": (
                sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else None),
            "selected_logical_cpu": selected_cpu,
            "selected_cpu_scaling_governor": (
                None if selected_cpu is None else read_optional(
                    f"/sys/devices/system/cpu/cpu{selected_cpu}/cpufreq/"
                    "scaling_governor")),
            "selected_cpu_thread_siblings": (
                None if selected_cpu is None else read_optional(
                    f"/sys/devices/system/cpu/cpu{selected_cpu}/topology/"
                    "thread_siblings_list")),
            "smt_active": read_optional("/sys/devices/system/cpu/smt/active"),
            "numa_nodes_online": read_optional(
                "/sys/devices/system/node/online"),
            "cpus_online": read_optional("/sys/devices/system/cpu/online"),
        }
    return {"contract": "Apple QoS is validated in every child process"}


def filesystem_contract_record(host_system, paths, formal):
    records = []
    for label, path in paths:
        resolved = pathlib.Path(path).resolve()
        if host_system == "Linux":
            filesystem = command_text(
                ["stat", "-f", "-c", "%T", str(resolved)])
            if formal and filesystem not in {"ext2/ext3", "ext4"}:
                raise RuntimeError(
                    f"formal Linux evidence requires native ext4 storage; "
                    f"{label} is {filesystem}: {resolved}")
        elif host_system == "Darwin":
            filesystem = command_text(
                ["stat", "-f", "%T", str(resolved)])
            if formal and filesystem.lower() != "apfs":
                raise RuntimeError(
                    f"formal macOS evidence requires APFS storage; {label} "
                    f"is {filesystem}: {resolved}")
        else:
            filesystem = "windows-volume:" + resolved.anchor
        records.append({
            "label": label, "path": str(resolved),
            "filesystem": filesystem,
        })
    return {"gate": "PASS", "paths": records}


def bench_stat_command(bench_stat, executable, runs, margin, boost_dir,
                       comparable_pairs, not_comparable_pairs,
                       require_clean, required_telemetry_prefix):
    command = [
        sys.executable, str(bench_stat), str(executable),
        "-n", str(runs), "--peer", f"boost={boost_dir}",
        "--equivalence-margin-pct", str(margin),
    ]
    if require_clean:
        command.append("--require-clean")
    command.extend((
        "--require-telemetry-prefix", required_telemetry_prefix))
    for baseline, rival in comparable_pairs:
        command.extend(("--pair", baseline, rival))
    for baseline, rival in not_comparable_pairs:
        command.extend(("--not-comparable-pair", baseline, rival))
    return command


def validate_log(text, path, runs, contracts, required_telemetry):
    if "(no rows matched pair:" in text:
        raise RuntimeError(f"pair manifest did not match rows: {path}")
    telemetry_prefix, expected_line = required_telemetry
    try:
        telemetry_records = [
            json.loads(line[len(CHILD_TELEMETRY_PREFIX):])
            for line in text.splitlines()
            if line.startswith(CHILD_TELEMETRY_PREFIX)
        ]
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"malformed child telemetry in {path}: {error}") from error
    if (len(telemetry_records) != runs or
            sorted(record.get("run") for record in telemetry_records) !=
            list(range(1, runs + 1)) or
            any(record.get("record") != "hyle_bench_child_telemetry" or
                record.get("schema") != 1 or
                record.get("prefix") != telemetry_prefix or
                record.get("line") != expected_line
                for record in telemetry_records)):
        raise RuntimeError(
            f"runtime child telemetry mismatch in {path}: expected exactly "
            f"one {expected_line!r} record for each of {runs} runs")
    records = [
        json.loads(line[len(RAW_PREFIX):])
        for line in text.splitlines() if line.startswith(RAW_PREFIX)
    ]
    expected_count = runs * len(contracts)
    if len(records) != expected_count:
        raise RuntimeError(
            f"raw record mismatch in {path}: got {len(records)}, "
            f"expected {expected_count}")
    expected = {
        (title, baseline, rival): comparable
        for title, baseline, rival, comparable in contracts
    }
    if len(expected) != len(contracts):
        raise RuntimeError("duplicate benchmark contracts in runner")
    grouped = {}
    for record in records:
        if (record.get("record") != "hyle_bench_pair_process" or
                record.get("schema") != 1):
            raise RuntimeError(f"raw record schema mismatch in {path}")
        contract = (record.get("title"), record.get("baseline"),
                    record.get("rival"))
        if contract not in expected:
            raise RuntimeError(
                f"unexpected title/pair contract in {path}: {contract}")
        if record.get("comparable") is not expected[contract]:
            raise RuntimeError(
                f"comparability mismatch in {path}: {contract}")
        baseline_ops = record.get("baseline_ops_per_second")
        rival_ops = record.get("rival_ops_per_second")
        ratio = record.get("ratio_rival_over_baseline")
        if (not isinstance(baseline_ops, (int, float)) or
                not isinstance(rival_ops, (int, float)) or
                not isinstance(ratio, (int, float)) or
                not math.isfinite(baseline_ops) or baseline_ops <= 0.0 or
                not math.isfinite(rival_ops) or rival_ops <= 0.0 or
                not math.isfinite(ratio) or ratio <= 0.0 or
                not math.isclose(ratio, rival_ops / baseline_ops,
                                 rel_tol=1e-12, abs_tol=0.0)):
            raise RuntimeError(f"invalid raw ratio in {path}: {contract}")
        pair = (record.get("baseline"), record.get("rival"))
        grouped.setdefault(pair, []).append(record)
    expected_pairs = {(baseline, rival)
                      for _, baseline, rival, _ in contracts}
    if set(grouped) != expected_pairs:
        raise RuntimeError(f"incomplete pair set in {path}")
    for pair, pair_records in grouped.items():
        if sorted(record.get("run") for record in pair_records) != list(
                range(1, runs + 1)):
            raise RuntimeError(f"incomplete run ids in {path}: {pair}")
    return grouped


def load_bench_stat(path):
    spec = importlib.util.spec_from_file_location("hyle_bench_stat", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load benchmark statistics: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one_order_summary(bench_stat, records, margin, comparable=True):
    ratios = [record["ratio_rival_over_baseline"] for record in records]
    decision = bench_stat.paired_verdict(
        ratios, margin, comparable=comparable)
    gate = bench_stat.sota_noninferiority_gate(
        ratios, margin, comparable=comparable)
    interval = decision["interval"]
    return {
        "ratio_median": decision["median"],
        "ratio_min": min(ratios),
        "ratio_max": max(ratios),
        "median_confidence_interval": (
            None if interval is None else interval[:2]),
        "exact_verdict": decision["verdict"],
        "exact_reason": decision["reason"],
        "sota_gate": gate["gate"],
        "sota_reason": gate["reason"],
    }


def dual_order_summary(bench_stat, grouped, pair, margin, comparable=True):
    first = one_order_summary(
        bench_stat, grouped["candidate-first"][pair], margin, comparable)
    second = one_order_summary(
        bench_stat, grouped["peer-first"][pair], margin, comparable)
    return {
        "candidate_first": first,
        "peer_first": second,
        "dual_order_exact_verdict": bench_stat.combine_order_verdicts(
            first["exact_verdict"], second["exact_verdict"],
            comparable=comparable, shape_stable=True),
        "dual_order_sota_gate": bench_stat.combine_sota_gates(
            first["sota_gate"], second["sota_gate"],
            comparable=comparable),
    }


def summarize_product(bench_stat, grouped_shapes, margin, placement):
    def shape_summary(grouped_allocations, pair):
        allocations = {
            allocation: dual_order_summary(
                bench_stat, grouped_allocations[allocation], pair, margin)
            for allocation, _ in ALLOCATION_STRATA
        }
        exact = None
        gate = None
        for allocation, _ in ALLOCATION_STRATA:
            summary = allocations[allocation]
            exact = (summary["dual_order_exact_verdict"] if exact is None
                     else bench_stat.combine_order_verdicts(
                         exact, summary["dual_order_exact_verdict"],
                         shape_stable=True))
            gate = (summary["dual_order_sota_gate"] if gate is None
                    else bench_stat.combine_sota_gates(
                        gate, summary["dual_order_sota_gate"]))
        return {
            "allocation_strata": allocations,
            "allocation_combined_exact_verdict": exact,
            "allocation_combined_sota_gate": gate,
        }

    rows = []
    for pair in PRODUCT_PAIRS:
        primary = shape_summary(grouped_shapes["primary"], pair)
        placement_summary = None
        if placement:
            placement_summary = shape_summary(
                grouped_shapes["placement"], pair)
            exact = bench_stat.combine_order_verdicts(
                primary["allocation_combined_exact_verdict"],
                placement_summary["allocation_combined_exact_verdict"],
                shape_stable=True)
            gate = bench_stat.combine_sota_gates(
                primary["allocation_combined_sota_gate"],
                placement_summary["allocation_combined_sota_gate"])
        else:
            exact = primary["allocation_combined_exact_verdict"]
            gate = primary["allocation_combined_sota_gate"]
        rows.append({
            "baseline": pair[0], "rival": pair[1],
            "primary": primary, "placement": placement_summary,
            "platform_exact_verdict": exact,
            "platform_sota_noninferiority_gate": gate,
        })
    return rows


def fastest_product_median_ns(grouped_shapes):
    medians = []
    for allocations in grouped_shapes.values():
        for orders in allocations.values():
            for pairs in orders.values():
                for records in pairs.values():
                    medians.append(statistics.median(
                        1e9 / record["baseline_ops_per_second"]
                        for record in records))
                    medians.append(statistics.median(
                        1e9 / record["rival_ops_per_second"]
                        for record in records))
    if not medians:
        raise RuntimeError("no product samples for floor budget")
    return min(medians)


def qualified_known_delta_gate(deltas, margin):
    margin_fraction = margin / 100.0
    near_ratio_upper = 1.0 / (1.0 + margin_fraction)
    near_ratio_lower = 1.0 / (
        1.0 + NEAR_DELTA_MAX_MARGIN_MULTIPLE * margin_fraction)
    if len(deltas) != 2:
        return False, near_ratio_lower, near_ratio_upper
    qualified = True
    for index, summary in enumerate(deltas):
        for order in ("candidate_first", "peer_first"):
            record = summary.get(order, {})
            interval = record.get("median_confidence_interval")
            median = record.get("ratio_median")
            if (record.get("exact_verdict") != "WIN" or
                    not isinstance(interval, (list, tuple)) or
                    len(interval) != 2 or
                    not all(isinstance(value, (int, float)) and
                            math.isfinite(value) for value in interval) or
                    not isinstance(median, (int, float)) or
                    not math.isfinite(median) or
                    interval[1] >= near_ratio_upper):
                qualified = False
            if (index == 0 and
                    (not isinstance(interval, (list, tuple)) or
                     len(interval) != 2 or
                     interval[0] < near_ratio_lower or
                     not isinstance(median, (int, float)) or
                     median < near_ratio_lower)):
                qualified = False
            if (index == 1 and
                    (not isinstance(interval, (list, tuple)) or
                     len(interval) != 2 or
                     interval[0] < LARGE_DELTA_RATIO_MIN or
                     interval[1] > LARGE_DELTA_RATIO_MAX or
                     not isinstance(median, (int, float)) or
                     not LARGE_DELTA_RATIO_MIN <= median <=
                     LARGE_DELTA_RATIO_MAX)):
                qualified = False
    for order in ("candidate_first", "peer_first"):
        delta_near = deltas[0].get(order, {}).get("ratio_median")
        delta_large = deltas[1].get(order, {}).get("ratio_median")
        if (not isinstance(delta_near, (int, float)) or
                not isinstance(delta_large, (int, float)) or
                not math.isfinite(delta_near) or
                not math.isfinite(delta_large) or
                not 0.0 < delta_large < delta_near < 1.0):
            qualified = False
    return qualified, near_ratio_lower, near_ratio_upper


def summarize_controls(bench_stat, grouped, grouped_shapes, margin):
    timer_floor = dual_order_summary(
        bench_stat, grouped, TIMER_FLOOR_PAIR, margin, comparable=False)
    phase_floor = dual_order_summary(
        bench_stat, grouped, PHASE_FLOOR_PAIR, margin, comparable=False)
    aa = dual_order_summary(bench_stat, grouped, AA_PAIR, margin)
    deltas = [
        dual_order_summary(bench_stat, grouped, pair, margin)
        for pair in DELTA_PAIRS
    ]
    aa_pass = aa["dual_order_exact_verdict"] == "EQUIVALENT"
    delta_pass, near_ratio_lower, near_ratio_upper = \
        qualified_known_delta_gate(deltas, margin)

    floor_ns = {"empty_timer": [], "two_phase_observers": []}
    for order in ("candidate-first", "peer-first"):
        for label, pair in (("empty_timer", TIMER_FLOOR_PAIR),
                            ("two_phase_observers", PHASE_FLOOR_PAIR)):
            for record in grouped[order][pair]:
                floor_ns[label].extend((
                    1e9 / record["baseline_ops_per_second"],
                    1e9 / record["rival_ops_per_second"],
                ))
    fastest_product = fastest_product_median_ns(grouped_shapes)
    floor_max = max(max(values) for values in floor_ns.values())
    floor_fraction = floor_max / fastest_product
    floor_pass = floor_fraction <= 0.01
    return {
        "empty_aggregate_floor": timer_floor,
        "phase_observer_floor": phase_floor,
        "empty_floor_max_ns_per_logical_op": max(floor_ns["empty_timer"]),
        "phase_observer_floor_max_ns_per_logical_op": max(
            floor_ns["two_phase_observers"]),
        "fastest_product_median_ns_per_logical_op": fastest_product,
        "max_fixed_floor_fraction_of_fastest_product": floor_fraction,
        "fixed_floor_budget_fraction": 0.01,
        "aa": aa,
        "known_delta": deltas,
        "near_delta_ratio_window": {
            "minimum_ratio_inclusive": near_ratio_lower,
            "maximum_ratio_exclusive": near_ratio_upper,
            "meaning": "entire CI proves a 3%-to-6% duration slowdown",
        },
        "large_delta_ratio_window": {
            "minimum_ratio_inclusive": LARGE_DELTA_RATIO_MIN,
            "maximum_ratio_inclusive": LARGE_DELTA_RATIO_MAX,
            "meaning": "entire CI brackets the declared 80/64 work control",
        },
        "aa_gate": "PASS" if aa_pass else "FAIL",
        "known_delta_gate": "PASS" if delta_pass else "FAIL",
        "fixed_floor_gate": "PASS" if floor_pass else "FAIL",
        "instrument_gate": (
            "PASS" if aa_pass and delta_pass and floor_pass else "FAIL"),
    }


def pe_text_section(tool, executable):
    command = [str(tool), "--section-headers", str(executable)]
    output = command_text(command)
    match = re.search(
        r"^\s*\d+\s+\.text\s+([0-9A-Fa-f]+)\s+"
        r"([0-9A-Fa-f]+)\s+TEXT\s*$", output, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot read PE .text section: {executable}")
    return {
        "path": str(executable), "sha256": sha256(executable),
        "text_size_bytes": int(match.group(1), 16),
        "text_vma": int(match.group(2), 16), "command": command,
    }


def normalized_object_code(tool, path):
    command = [str(tool), "--disassemble", "--reloc", str(path)]
    lines = command_text(command).splitlines()
    if lines and "file format" in lines[0]:
        lines[0] = "file format" + lines[0].split("file format", 1)[1]
    text = "\n".join(lines) + "\n"
    return {
        "path": str(path), "object_sha256": sha256(path),
        "normalized_disassembly_sha256": hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        "command": command,
    }


def _formal_artifact_architecture_matches(host_system, text):
    lowered = text.lower()
    format_patterns = {
        "Windows": ("file format coff-x86-64",),
        "Linux": ("file format elf64-x86-64",),
        "Darwin": ("file format mach-o arm64",),
    }[host_system]
    architecture_patterns = {
        "Windows": (r"^architecture:\s*x86_64\s*$",),
        "Linux": (r"^architecture:\s*x86_64\s*$",),
        "Darwin": (r"^architecture:\s*(?:aarch64|arm64)\s*$",),
    }[host_system]
    return (any(pattern in lowered for pattern in format_patterns) and
            any(re.search(pattern, lowered, re.MULTILINE)
                for pattern in architecture_patterns))


def verify_artifact_architecture(tool, host_system, artifacts, output_dir):
    records = []
    for label, path in artifacts:
        command = [str(tool), "--file-headers", str(path)]
        output = command_text(command)
        if not _formal_artifact_architecture_matches(host_system, output):
            raise RuntimeError(
                f"formal {host_system} artifact architecture mismatch for "
                f"{label}: {path}\n{output}")
        records.append({
            "label": label,
            "path": str(path),
            "sha256": sha256(path),
            "command": command,
            "file_headers_sha256": hashlib.sha256(
                output.encode("utf-8")).hexdigest(),
        })
    record = {
        "gate": "PASS",
        "expected_host_artifact": {
            "Windows": "COFF x86-64",
            "Linux": "ELF x86-64",
            "Darwin": "Mach-O arm64",
        }[host_system],
        "artifacts": records,
    }
    path = output_dir / "artifact-architecture.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    record["record_path"] = str(path)
    record["record_sha256"] = sha256(path)
    return record


def _split_disassembly_operands(text):
    operands = []
    start = 0
    depth = 0
    for index, character in enumerate(text):
        if character in "([":
            depth += 1
        elif character in ")]":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            operands.append(text[start:index].strip())
            start = index + 1
    operands.append(text[start:].strip())
    return [operand for operand in operands if operand]


def _canonical_disassembly_register(text):
    register = text.strip().lower().lstrip("%")
    if re.fullmatch(r"[er](ax|bx|cx|dx|si|di|bp|sp)", register):
        return register[1:]
    match = re.fullmatch(r"r(\d+)[dwb]?", register)
    if match:
        return "r" + match.group(1)
    match = re.fullmatch(r"w(\d+)", register)
    if match:
        return "x" + match.group(1)
    if register == "wzr":
        return "xzr"
    return register


def _disassembly_memory_operand(text):
    return "(" in text or "[" in text


def _exact_zero_immediate(text):
    # Match the complete immediate token. A prefix match here used to classify
    # `cmp $0x1000, %r8` as a comparison with zero and allowed a construction
    # loop to masquerade as the linked-node retirement loop.
    token = text.split("# imm", 1)[0].strip().lower()
    return token in {"$0", "$0x0", "#0", "#0x0"}


def _disassembly_control_transfer(mnemonic):
    return (
        mnemonic.startswith(("j", "call", "ret", "loop")) or
        mnemonic in {"b", "bl", "blr", "br", "cbz", "cbnz",
                     "tbz", "tbnz"} or
        mnemonic.startswith("b.")
    )


def _disassembly_loop_branch(mnemonic):
    # AArch64 `bl`/`blr` are calls, not loop backedges. Treating their target
    # as a branch once allowed an unrelated backward helper call to masquerade
    # as the teardown loop.
    return (
        mnemonic.startswith(("j", "loop")) or
        mnemonic == "b" or mnemonic.startswith("b.") or
        mnemonic in {"cbz", "cbnz", "tbz", "tbnz"}
    )


def _disassembly_instruction_match(line):
    return re.match(
        r"^\s*([0-9A-Fa-f]+):\s+"
        r"(?:(?:[0-9A-Fa-f]{2}\s+)+|"
        r"[0-9A-Fa-f]{8}\s+)\s*"
        r"([A-Za-z][A-Za-z0-9.]*)\s*(.*)$", line)


def _written_disassembly_registers(mnemonic, operands):
    """Conservatively identify explicit register destinations.

    llvm-objdump prints x86 in AT&T source,destination order and AArch64 in
    destination,source order. This kill set prevents a register that was once
    proven zero from remaining zero after a later overwrite.
    """
    if not operands:
        return set()
    x86_att = any("%" in operand for operand in operands)
    read_only = (
        mnemonic.startswith(("cmp", "test", "tst", "str", "stur", "stp"))
        or _disassembly_control_transfer(mnemonic)
        or mnemonic.startswith("nop")
        or mnemonic.startswith("push")
    )
    if read_only:
        return set()
    if x86_att:
        destination = operands[-1]
        if _disassembly_memory_operand(destination):
            return set()
        register = _canonical_disassembly_register(destination)
        return {register} if re.fullmatch(
            r"(?:[abcd]x|[sb]p|[sd]i|r\d+)", register) else set()
    # AArch64 pair loads have two explicit destinations; ordinary data and load
    # instructions write their first operand.
    count = 2 if mnemonic.startswith("ldp") else 1
    result = set()
    for destination in operands[:count]:
        if _disassembly_memory_operand(destination):
            continue
        register = _canonical_disassembly_register(destination)
        if re.fullmatch(r"x(?:\d+|zr)", register):
            result.add(register)
    return result


def _call_clobbered_registers(mnemonic, host_system):
    if not (mnemonic.startswith("call") or mnemonic in {"bl", "blr"}):
        return set()
    if host_system == "Windows":
        x86 = {"ax", "cx", "dx", *("r" + str(i) for i in range(8, 12))}
    else:
        x86 = {
            "ax", "cx", "dx", "si", "di",
            *("r" + str(i) for i in range(8, 12)),
        }
    arm = {"x" + str(i) for i in range(19)}
    return x86 | arm


def _analyze_lifetime_function(name, text, host_system=None):
    """Read one demangled function conservatively enough for a DSE gate.

    Zero-register facts are intentionally less powerful than a general CFG
    analysis: only definitions in the straight-line entry preheader are
    accepted. Such a definition cannot be skipped by a branch. Any later
    explicit overwrite or ABI call-clobber permanently removes the fact.
    Direct zero-immediate stores and AArch64 xzr remain independently visible.
    """
    if host_system is None:
        host_system = platform.system()
    instructions = []
    zero_registers = {"xzr"}
    in_entry_preheader = True
    for line in text.splitlines()[1:]:
        match = _disassembly_instruction_match(line)
        if match is None:
            continue
        mnemonic = match.group(2).lower()
        operand_text = match.group(3)
        operands = _split_disassembly_operands(operand_text)
        canonical = [
            _canonical_disassembly_register(operand) for operand in operands
        ]

        zero_registers.difference_update(
            _written_disassembly_registers(mnemonic, operands))
        zero_registers.difference_update(
            _call_clobbered_registers(mnemonic, host_system))
        zero_registers.add("xzr")
        x86_att = any("%" in operand for operand in operands)
        zero_destination = None
        if (mnemonic.startswith(("xor", "pxor")) and
                len(canonical) >= 2 and
                canonical[-2] == canonical[-1]):
            zero_destination = canonical[-1]
        elif (mnemonic.startswith("eor") and len(canonical) >= 3 and
              canonical[1] == canonical[2]):
            zero_destination = canonical[0]
        elif mnemonic.startswith("mov") and len(operands) == 2:
            if x86_att and _exact_zero_immediate(operands[0]):
                zero_destination = canonical[1]
            elif (not x86_att and
                  (_exact_zero_immediate(operands[1]) or
                   canonical[1] == "xzr")):
                zero_destination = canonical[0]
        if in_entry_preheader and zero_destination is not None:
            zero_registers.add(zero_destination)

        is_store = False
        source_operands = []
        store_words = 0
        if mnemonic.startswith(("str", "stur", "stp")) and operands:
            memory_indices = [
                index for index, operand in enumerate(operands)
                if _disassembly_memory_operand(operand)
            ]
            if memory_indices:
                # Post-index AArch64 stores end in an immediate operand, e.g.
                # `stp xzr, xzr, [x8], #16`. Sources are everything before
                # the first address operand, not blindly `operands[:-1]`.
                memory_index = memory_indices[0]
                is_store = True
                source_operands = operands[:memory_index]
                store_words = 2 if mnemonic.startswith("stp") else 1
        elif (mnemonic.startswith(("mov", "vmov")) and
              len(operands) >= 2 and
              _disassembly_memory_operand(operands[-1])):
            is_store = True
            source_operands = operands[:-1]
            store_words = (
                2 if any(token in mnemonic for token in (
                    "ups", "aps", "dqu", "dqa")) else 1)
        null_store_words = 0
        if is_store:
            zero_sources = sum(
                _canonical_disassembly_register(source) in zero_registers or
                _exact_zero_immediate(source)
                for source in source_operands)
            if zero_sources:
                null_store_words = min(store_words, zero_sources)
                if store_words == 2 and len(source_operands) == 1:
                    null_store_words = 2

        is_load = False
        if mnemonic.startswith(("ldr", "ldur", "ldp")):
            is_load = any(
                _disassembly_memory_operand(operand) for operand in operands)
        elif (mnemonic.startswith(("mov", "vmov")) and
              len(operands) >= 2 and
              _disassembly_memory_operand(operands[0]) and
              not _disassembly_memory_operand(operands[-1])):
            is_load = True
        instructions.append({
            "address": int(match.group(1), 16),
            "mnemonic": mnemonic,
            "operands": operand_text,
            "is_memory_load": is_load,
            "store_words": store_words,
            "null_store_words": null_store_words,
            "known_zero_registers": sorted(zero_registers),
        })
        if _disassembly_control_transfer(mnemonic):
            in_entry_preheader = False

    loops = []
    for branch in instructions:
        if not _disassembly_loop_branch(branch["mnemonic"]):
            continue
        # Demangled target symbols contain template commas, so isolate the
        # machine operand prefix before splitting. For x86 jcc the target is
        # the only operand; for AArch64 cb/tb it is the final operand.
        operand_prefix = branch["operands"].split("<", 1)[0]
        operand_prefix = operand_prefix.split("# imm", 1)[0]
        target_operand = operand_prefix.rsplit(",", 1)[-1].strip()
        target = re.fullmatch(r"(?:0x)?([0-9A-Fa-f]+)", target_operand)
        if target is None:
            continue
        target_address = int(target.group(1), 16)
        if target_address >= branch["address"]:
            continue
        body = [
            instruction for instruction in instructions
            if target_address <= instruction["address"] <= branch["address"]
        ]
        loops.append({
            "target": target_address,
            "branch": branch["address"],
            "instruction_count": len(body),
            "memory_load_count": sum(
                instruction["is_memory_load"] for instruction in body),
            "store_words": sum(
                instruction["store_words"] for instruction in body),
            "null_store_words": sum(
                instruction["null_store_words"] for instruction in body),
            "has_linked_pointer_test": any(
                instruction["mnemonic"].startswith(("test", "tst")) or
                instruction["mnemonic"].startswith(("cbz", "cbnz")) or
                (instruction["mnemonic"].startswith("cmp") and
                 any(_exact_zero_immediate(operand)
                     for operand in _split_disassembly_operands(
                         instruction["operands"])))
                for instruction in body),
        })
    observer_symbol = "hyle_container_intrusive_observe_storage_phase"
    observer_addresses = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if observer_symbol not in line:
            continue
        for candidate in reversed(lines[:index + 1]):
            instruction = _disassembly_instruction_match(candidate)
            if instruction is not None:
                mnemonic = instruction.group(2).lower()
                if mnemonic.startswith("call") or mnemonic == "bl":
                    observer_addresses.append(
                        int(instruction.group(1), 16))
                break
    observer_addresses = sorted(set(observer_addresses))
    return {
        "symbol": name,
        "disassembly_sha256": hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        "instruction_count": len(instructions),
        "backward_loops": loops,
        "observer_reference": bool(observer_addresses),
        "observer_call_addresses": observer_addresses,
    }


def _last_loop_before_observer(record):
    addresses = record["observer_call_addresses"]
    if len(addresses) != 2:
        raise RuntimeError(
            "lifetime function must contain exactly two phase observers")
    linked_observer, final_observer = sorted(addresses)
    teardown_loops = [
        loop for loop in record["backward_loops"]
        if linked_observer < loop["branch"] < final_observer
    ]
    return (max(teardown_loops, key=lambda loop: loop["branch"])
            if teardown_loops else None)


def lifetime_workload_codegen(tool, path, output_dir, artifact_label):
    command = [
        str(tool), "--disassemble", "--reloc", "--demangle", str(path)]
    disassembly = command_text(command)
    disassembly_path = output_dir / (
        f"lifetime-workload-{artifact_label}-disassembly.txt")
    disassembly_path.write_text(disassembly + "\n", encoding="utf-8")
    headers = list(re.finditer(
        r"^[0-9A-Fa-f]+\s+<(.+)>:\s*$", disassembly, re.MULTILINE))

    def find_function(tokens, forbidden=()):
        matches = []
        for index, header in enumerate(headers):
            name = header.group(1)
            if ("nanobench" not in name and "dtor" not in name and
                    all(token in name for token in tokens) and
                    not any(token in name for token in forbidden)):
                end = (headers[index + 1].start()
                       if index + 1 < len(headers) else len(disassembly))
                matches.append((name, disassembly[header.start():end]))
        if len(matches) != 1:
            raise RuntimeError(
                "lifetime codegen symbol match is not unique for " +
                repr(tokens) + f": {len(matches)}")
        return _analyze_lifetime_function(*matches[0])

    records = {
        "hyle_auto_owner_teardown": find_function(
            ("hyle_owner_teardown_workload", "Node"), ("SafeNode",)),
        "hyle_safe_owner_teardown": find_function(
            ("hyle_owner_teardown_workload", "SafeNode")),
        "boost_auto_owner_teardown": find_function(
            ("boost_owner_teardown_workload", "BoostNode"),
            ("BoostSafeNode",)),
        "boost_safe_owner_teardown": find_function(
            ("boost_owner_teardown_workload", "BoostSafeNode")),
        "hyle_auto_node_retirement": find_function(
            ("hyle_auto_node_retirement_workload",)),
        "boost_auto_node_retirement": find_function(
            ("boost_auto_node_retirement_workload",)),
    }
    for name in (
            "hyle_auto_owner_teardown", "hyle_safe_owner_teardown",
            "boost_auto_owner_teardown", "boost_safe_owner_teardown"):
        record = records[name]
        try:
            selected = _last_loop_before_observer(record)
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        record["selected_last_loop_before_observer"] = selected
        qualified = (
            selected is not None and selected["memory_load_count"] >= 1 and
            selected["store_words"] >= 2)
        if not record["observer_reference"] or not qualified:
            raise RuntimeError(
                f"{name} contains no next-load plus two-store O(N) clear "
                "loop between linked/unlinked phase observers; "
                f"selected={selected!r}")
    for name in (
            "hyle_auto_node_retirement", "boost_auto_node_retirement"):
        record = records[name]
        try:
            selected = _last_loop_before_observer(record)
        except RuntimeError as error:
            raise RuntimeError(f"{name}: {error}") from error
        record["selected_last_loop_before_observer"] = selected
        qualified = (
            selected is not None and selected["has_linked_pointer_test"] and
            selected["memory_load_count"] >= 1 and
            selected["store_words"] >= 4)
        if not record["observer_reference"] or not qualified:
            raise RuntimeError(
                f"{name} contains no linked-test, load and four-store O(N) "
                "retirement loop between phase observers; "
                f"selected={selected!r}")
    record = {
        "gate": "PASS",
        "claim": (
            "two opaque storage phases retain O(N) owner clear and "
            "linked-node auto-unlink destructor loops"),
        "object_path": str(path),
        "object_sha256": sha256(path),
        "command": command,
        "disassembly_path": str(disassembly_path),
        "disassembly_file_sha256": sha256(disassembly_path),
        "functions": records,
    }
    output_path = output_dir / (
        f"lifetime-workload-{artifact_label}-codegen.json")
    output_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    record["record_path"] = str(output_path)
    record["record_sha256"] = sha256(output_path)
    return record


def verify_no_interprocedural_optimization(build, output_dir):
    candidates = [build / "CMakeCache.txt", build / "build.ninja"]
    candidates.extend(build.rglob("hyle_container_intrusive_bench.vcxproj"))
    candidates.extend(build.rglob(
        "hyle_container_intrusive_bench_placement.vcxproj"))
    candidates.extend(build.rglob(
        "hyle_container_intrusive_instrument_bench.vcxproj"))
    candidates.extend(build.rglob("flags.make"))
    inspected = []
    forbidden = re.compile(
        r"(?:/GL(?:\s|<)|-flto(?:[=\s]|$)|"
        r"<WholeProgramOptimization>true)", re.IGNORECASE)
    for path in sorted(set(path for path in candidates if path.is_file())):
        text = path.read_text(encoding="utf-8", errors="replace")
        if forbidden.search(text):
            raise RuntimeError(
                "container_intrusive lifetime observer requires IPO/LTO off; "
                f"forbidden option found in {path}")
        inspected.append({"path": str(path), "sha256": sha256(path)})
    if not inspected:
        raise RuntimeError("no build metadata was available for the no-IPO gate")
    record = {
        "gate": "PASS",
        "claim": (
            "separate lifetime observer translation unit remains opaque to "
            "benchmark callbacks"),
        "forbidden_options": ["/GL", "-flto", "WholeProgramOptimization"],
        "inspected_files": inspected,
    }
    path = output_dir / "no-interprocedural-optimization.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    record["record_path"] = str(path)
    record["record_sha256"] = sha256(path)
    return record


def _msvc_release_shape_errors(fields):
    errors = []
    options = fields.get("AdditionalOptions", "")
    definitions = fields.get("PreprocessorDefinitions", "")
    if re.search(
            r"(?:^|[\s;])/(?:Od|O1|Ob0|RTC\w*|GL)(?=$|[\s;])",
            options, re.IGNORECASE):
        errors.append("forbidden compiler option")
    if re.search(
            r"(?:^|[\s;/])(?:/D)?_DEBUG(?:[=;\s]|$)",
            options + ";" + definitions, re.IGNORECASE):
        errors.append("_DEBUG")
    iterator_debug = re.search(
        r"(?:^|[\s;/])(?:/D)?_ITERATOR_DEBUG_LEVEL(?:=|#)(\d+)",
        options + ";" + definitions, re.IGNORECASE)
    if iterator_debug is not None and int(iterator_debug.group(1)) != 0:
        errors.append("nonzero _ITERATOR_DEBUG_LEVEL")
    if fields.get("BasicRuntimeChecks", "") not in {"", "Default"}:
        errors.append("BasicRuntimeChecks")
    if "Debug" in fields.get("RuntimeLibrary", ""):
        errors.append("debug runtime library")
    return errors


def _unix_release_shape_errors(flags, definitions):
    errors = []
    tokens = shlex.split(flags) + shlex.split(definitions)
    forbidden_exact = {"-pg", "--coverage"}
    forbidden_prefixes = (
        "-fno-inline", "-fno-builtin", "-fno-tree-", "-fno-ipa-",
        "-fsanitize", "-fprofile-", "-finstrument-functions",
    )
    for token in tokens:
        if token in forbidden_exact or token.startswith(forbidden_prefixes):
            errors.append(token)
        if token == "-D_DEBUG" or token.startswith(
                ("-D_GLIBCXX_DEBUG", "-D_GLIBCXX_ASSERTIONS")):
            errors.append(token)
        match = re.fullmatch(r"-D_ITERATOR_DEBUG_LEVEL(?:=|#)(\d+)", token)
        if match is not None and int(match.group(1)) != 0:
            errors.append(token)
    return errors


def verify_release_optimization(build, host_system, placement, output_dir):
    target_names = [
        "hyle_container_intrusive_bench",
        "hyle_container_intrusive_instrument_bench",
    ]
    if placement:
        target_names.append("hyle_container_intrusive_bench_placement")
    if host_system == "Windows":
        records = []
        for target in target_names:
            matches = list(build.rglob(target + ".vcxproj"))
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one MSVC project for {target}; "
                    f"found {len(matches)}")
            path = matches[0]
            text = path.read_text(encoding="utf-8", errors="replace")
            root = ET.fromstring(text)

            def local_name(element):
                return element.tag.rsplit("}", 1)[-1]

            groups = [
                element for element in root.iter()
                if local_name(element) == "ItemDefinitionGroup" and
                "Release|" in element.attrib.get("Condition", "")
            ]
            if len(groups) != 1:
                raise RuntimeError(
                    f"expected one Release item-definition group in {path}")
            compile_nodes = [
                element for element in groups[0]
                if local_name(element) == "ClCompile"
            ]
            if len(compile_nodes) != 1:
                raise RuntimeError(f"missing Release ClCompile in {path}")
            fields = {
                local_name(element): (element.text or "")
                for element in compile_nodes[0]
            }
            additional = fields.get("AdditionalOptions", "")
            if (fields.get("Optimization") != "MaxSpeed" or
                    fields.get("InlineFunctionExpansion") !=
                    "AnySuitable" or
                    "NDEBUG" not in fields.get(
                        "PreprocessorDefinitions", "") or
                    _msvc_release_shape_errors(fields)):
                raise RuntimeError(
                    f"formal MSVC Release optimization contract failed: "
                    f"{path}")
            source_items = [
                element for element in root.iter()
                if local_name(element) == "ClCompile" and
                "Include" in element.attrib
            ]
            source_override_records = []
            for source in source_items:
                applicable = {}
                for field in source:
                    condition = field.attrib.get("Condition", "")
                    if condition and "Release|" not in condition:
                        continue
                    applicable[local_name(field)] = field.text or ""
                source_options = applicable.get("AdditionalOptions", "")
                if (applicable.get("Optimization") not in
                        {None, "", "MaxSpeed"} or
                        applicable.get("InlineFunctionExpansion") in
                        {"Disabled", "OnlyExplicitInline"} or
                        _msvc_release_shape_errors(applicable)):
                    raise RuntimeError(
                        "formal MSVC per-source Release override failed: "
                        f"{source.attrib['Include']}")
                source_override_records.append({
                    "source": source.attrib["Include"],
                    "release_override_sha256": hashlib.sha256(
                        json.dumps(applicable, sort_keys=True).encode(
                            "utf-8")).hexdigest(),
                })
            records.append({
                "target": target,
                "path": str(path),
                "sha256": sha256(path),
                "optimization": fields.get("Optimization"),
                "inline_function_expansion": fields.get(
                    "InlineFunctionExpansion"),
                "release_definitions_sha256": hashlib.sha256(
                    fields.get("PreprocessorDefinitions", "").encode(
                        "utf-8")).hexdigest(),
                "additional_options_sha256": hashlib.sha256(
                    additional.encode("utf-8")).hexdigest(),
                "source_release_overrides": source_override_records,
            })
    else:
        path = build / "build.ninja"
        if not path.is_file():
            raise RuntimeError(
                "formal Linux/macOS optimization gate requires Ninja")
        text = path.read_text(encoding="utf-8", errors="replace")
        records = []
        for target in target_names:
            marker = target + ".dir/"
            blocks = []
            lines = text.splitlines()
            for index, line in enumerate(lines):
                if not line.startswith("build ") or marker not in line:
                    continue
                block = [line]
                cursor = index + 1
                while cursor < len(lines) and (
                        lines[cursor].startswith("  ") or
                        not lines[cursor]):
                    block.append(lines[cursor])
                    cursor += 1
                blocks.append("\n".join(block))
            if not blocks:
                raise RuntimeError(
                    f"no Ninja object compile blocks found for {target}")
            for block in blocks:
                flags_match = re.search(
                    r"^  FLAGS = (.*)$", block, re.MULTILINE)
                definitions_match = re.search(
                    r"^  DEFINES = (.*)$", block, re.MULTILINE)
                if flags_match is None:
                    raise RuntimeError(
                        f"missing compile flags for {target} object")
                flags = shlex.split(flags_match.group(1))
                definitions = ("" if definitions_match is None else
                               definitions_match.group(1))
                optimizations = [
                    flag for flag in flags
                    if re.fullmatch(r"-O(?:0|1|2|3|g|s|fast)", flag)
                ]
                if (not optimizations or
                        optimizations[-1] not in {"-O2", "-O3", "-Ofast"} or
                        any(flag in {"-O0", "-O1", "-Og"}
                            for flag in optimizations) or
                        not re.search(r"(?:^|\s)-DNDEBUG(?:\s|$)", block) or
                        _unix_release_shape_errors(
                            flags_match.group(1), definitions)):
                    raise RuntimeError(
                        f"formal {host_system} Release optimization "
                        f"contract failed for {target}")
                records.append({
                    "target": target,
                    "object_build_line": block.splitlines()[0],
                    "compile_block_sha256": hashlib.sha256(
                        block.encode("utf-8")).hexdigest(),
                    "effective_optimization": optimizations[-1],
                })
        records.append({"path": str(path), "sha256": sha256(path)})
    record = {
        "gate": "PASS",
        "contract": "Release optimized, NDEBUG, no debug optimization",
        "records": records,
    }
    path = output_dir / "release-optimization.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    record["record_path"] = str(path)
    record["record_sha256"] = sha256(path)
    return record


def object_text_section(tool, path):
    command = [str(tool), "--section-headers", str(path)]
    output = command_text(command)
    match = re.search(
        r"^\s*\d+\s+(?:\.text|__text)\s+([0-9A-Fa-f]+)\s+",
        output, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot read object text section: {path}")
    return {
        "path": str(path), "sha256": sha256(path),
        "text_size_bytes": int(match.group(1), 16), "command": command,
    }


def ninja_target_compile_records(build, target):
    path = build / "build.ninja"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    marker = target + ".dir/"
    records = {}
    for index, line in enumerate(lines):
        if not line.startswith("build ") or marker not in line:
            continue
        output = line[len("build "):].split(": ", 1)[0]
        name_match = re.search(r"/([^/]+\.cc)\.o$", output)
        if name_match is None:
            continue
        variables = {}
        cursor = index + 1
        while cursor < len(lines) and (
                lines[cursor].startswith("  ") or not lines[cursor]):
            variable = re.match(r"^  ([A-Z_]+) = (.*)$", lines[cursor])
            if variable is not None:
                variables[variable.group(1)] = variable.group(2)
            cursor += 1
        source = name_match.group(1)
        if source in records:
            raise RuntimeError(
                f"duplicate Ninja compile record for {target}/{source}")
        records[source] = variables
    if not records:
        raise RuntimeError(f"no Ninja compile records for {target}")
    return records


def verify_placement(host_system, tool, build, primary_executable,
                     placement_executable, primary_object, placement_object,
                     output_dir):
    if host_system == "Windows":
        primary = pe_text_section(tool, primary_executable)
        placement = pe_text_section(tool, placement_executable)
        if placement["text_size_bytes"] <= primary["text_size_bytes"]:
            raise RuntimeError("/FUNCTIONPADMIN did not increase PE .text size")
        primary_code = normalized_object_code(tool, primary_object)
        placement_code = normalized_object_code(tool, placement_object)
        if (primary_code["normalized_disassembly_sha256"] !=
                placement_code["normalized_disassembly_sha256"]):
            raise RuntimeError(
                "primary/placement compiler object code is not identical")
        record = {
            "mechanism": "MSVC LINK /FUNCTIONPADMIN:64",
            "contract_effect": "inter-function padding only",
            "candidate_and_peer_share_executable": True,
            "primary": primary, "placement": placement,
            "text_size_delta_bytes": (
                placement["text_size_bytes"] - primary["text_size_bytes"]),
            "primary_object": primary_code,
            "placement_object": placement_code,
            "object_code_identical_before_link_padding": True,
        }
    else:
        primary_records = ninja_target_compile_records(
            build, "hyle_container_intrusive_bench")
        placement_records = ninja_target_compile_records(
            build, "hyle_container_intrusive_bench_placement")
        if set(primary_records) != set(placement_records):
            raise RuntimeError(
                "primary/placement Ninja source sets differ")
        compile_records = []
        for source in sorted(primary_records):
            primary_variables = primary_records[source]
            placement_variables = placement_records[source]
            primary_flags = shlex.split(primary_variables.get("FLAGS", ""))
            placement_flags = shlex.split(
                placement_variables.get("FLAGS", ""))
            if "-falign-functions=64" in primary_flags or \
                    placement_flags.count("-falign-functions=64") != 1:
                raise RuntimeError(
                    "placement target did not isolate -falign-functions=64")
            placement_without_alignment = [
                flag for flag in placement_flags
                if flag != "-falign-functions=64"
            ]
            if (primary_flags != placement_without_alignment or
                    primary_variables.get("DEFINES", "") !=
                    placement_variables.get("DEFINES", "") or
                    primary_variables.get("INCLUDES", "") !=
                    placement_variables.get("INCLUDES", "")):
                raise RuntimeError(
                    f"placement compile contract differs beyond alignment: "
                    f"{source}")
            compile_records.append({
                "source": source,
                "primary_flags": primary_flags,
                "placement_flags": placement_flags,
            })
        primary = object_text_section(tool, primary_object)
        placement = object_text_section(tool, placement_object)
        if placement["text_size_bytes"] <= primary["text_size_bytes"]:
            raise RuntimeError(
                "-falign-functions=64 did not increase object text size")
        record = {
            "mechanism": "compiler -falign-functions=64",
            "contract_effect": "symmetric function-address alignment",
            "candidate_and_peer_share_executable": True,
            "primary": primary, "placement": placement,
            "text_size_delta_bytes": (
                placement["text_size_bytes"] - primary["text_size_bytes"]),
            "compile_contracts": compile_records,
            "only_compile_difference_is_function_alignment": True,
        }
    path = output_dir / "placement-layout.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    record["record_path"] = str(path)
    record["record_sha256"] = sha256(path)
    return record


def main():
    parser = argparse.ArgumentParser(
        description=(
            "legacy container_intrusive benchmark diagnostic; not a current "
            "module certificate"))
    parser.add_argument("--build", required=True)
    parser.add_argument("--boost-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="Release")
    parser.add_argument("--fresh-configure", action="store_true")
    parser.add_argument("--generator")
    parser.add_argument("--architecture")
    parser.add_argument("--llvm-objdump")
    parser.add_argument("--placement", action="store_true")
    parser.add_argument("--affinity-mask")
    parser.add_argument(
        "--apple-qos", choices=("interactive", "utility"))
    parser.add_argument(
        "--priority-class", default="normal",
        choices=("normal", "above-normal", "high"))
    parser.add_argument("-n", "--runs", type=int, default=63)
    parser.add_argument(
        "--equivalence-margin-pct", type=float, default=3.0)
    parser.add_argument("--prior-inconclusive-manifest")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    host_system = platform.system()
    if host_system not in {"Windows", "Linux", "Darwin"}:
        parser.error(f"unsupported formal benchmark host: {host_system}")

    if args.runs < 15:
        parser.error("formal evidence requires at least 15 process samples")
    if args.config.lower() != "release":
        parser.error("container_intrusive performance evidence requires Release")
    if (not math.isfinite(args.equivalence_margin_pct) or
            args.equivalence_margin_pct <= 0.0):
        parser.error("equivalence margin must be finite and positive")
    if args.require_clean:
        if not math.isclose(args.equivalence_margin_pct, FORMAL_MARGIN_PCT,
                            rel_tol=0.0, abs_tol=1e-12):
            parser.error("formal evidence fixes the non-inferiority margin at 3%")
        if host_system == "Windows" and args.runs != 63:
            parser.error("formal Windows evidence requires exactly n=63")
        if host_system in {"Linux", "Darwin"} and args.runs not in {15, 31}:
            parser.error("formal Linux/macOS evidence permits only n=15 or n=31")
        if (host_system in {"Linux", "Darwin"} and args.runs == 31 and
                args.prior_inconclusive_manifest is None):
            parser.error(
                "formal n=31 requires --prior-inconclusive-manifest from "
                "the complete n=15 experiment")
        if args.runs != 31 and args.prior_inconclusive_manifest is not None:
            parser.error(
                "--prior-inconclusive-manifest is only valid for formal n=31")
    if args.fresh_configure and args.skip_build:
        parser.error("--fresh-configure and --skip-build are incompatible")
    if (args.generator or args.architecture) and not args.fresh_configure:
        parser.error("--generator/--architecture require --fresh-configure")
    if args.placement and not args.llvm_objdump:
        parser.error("--placement requires --llvm-objdump")
    if args.require_clean and not args.llvm_objdump:
        parser.error(
            "formal evidence requires --llvm-objdump for the actual "
            "benchmark-object lifetime gate")
    if args.require_clean and not args.placement:
        parser.error("formal evidence requires --placement on every platform")
    if args.require_clean and not args.fresh_configure:
        parser.error(
            "formal clean evidence requires a fresh configure/build binding")
    if args.affinity_mask is not None:
        try:
            affinity_value = int(args.affinity_mask, 0)
        except ValueError:
            parser.error("--affinity-mask must be an integer such as 0x4")
    else:
        affinity_value = None
    if args.require_clean and host_system in {"Windows", "Linux"} and \
            args.affinity_mask is None:
        parser.error(
            "formal Windows/Linux evidence requires an explicit "
            "--affinity-mask")
    if (args.require_clean and affinity_value is not None and
            affinity_value.bit_count() != 1):
        parser.error("formal affinity must select exactly one logical CPU")
    if host_system == "Darwin" and args.affinity_mask is not None:
        parser.error("macOS has no supported hard CPU-affinity contract")
    if args.apple_qos is not None and host_system != "Darwin":
        parser.error("--apple-qos is only valid on macOS")
    if args.require_clean and host_system == "Windows" and \
            args.priority_class != "above-normal":
        parser.error("formal Windows evidence requires above-normal priority")
    if args.require_clean and host_system == "Darwin" and \
            args.apple_qos != "interactive":
        parser.error("formal macOS evidence requires interactive QoS")
    try:
        build_environment = build_environment_record(
            host_system, args.require_clean)
        host_identity = host_identity_record(args.require_clean)
    except RuntimeError as error:
        parser.error(str(error))

    repo = pathlib.Path(__file__).resolve().parents[1]
    build = pathlib.Path(args.build).resolve()
    boost_dir = pathlib.Path(args.boost_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    bench_stat_path = repo / "scripts" / "bench-stat.py"
    if not (boost_dir / "CMakeLists.txt").is_file():
        parser.error(f"Boost checkout is unavailable: {boost_dir}")
    source_git = git_record(repo)
    peer_git = git_record(boost_dir)
    if args.require_clean and (not source_git["clean"] or
                               not peer_git["clean"]):
        parser.error("--require-clean needs clean source and peer checkouts")
    if args.require_clean and peer_git["head"] != FORMAL_BOOST_HEAD:
        parser.error(
            f"formal peer must be Boost commit {FORMAL_BOOST_HEAD}")
    try:
        output_dir.relative_to(repo)
        output_inside_repo = True
    except ValueError:
        output_inside_repo = False
    if args.require_clean and output_inside_repo:
        parser.error("clean evidence output must be outside the source tree")
    try:
        build.relative_to(repo)
        build_inside_repo = True
    except ValueError:
        build_inside_repo = False
    if args.require_clean and build_inside_repo:
        parser.error("clean evidence build must be outside the source tree")
    if args.fresh_configure and build.exists():
        parser.error(f"fresh build tree already exists: {build}")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output directory already exists: {output_dir}")

    input_hashes_before = collect_input_hashes(repo)
    prior_inconclusive_record = None
    commands = []
    if args.fresh_configure:
        configure = [
            "cmake", "-S", str(repo), "-B", str(build),
            "-DHYLE_COMPONENTS=container_intrusive",
            "-DHYLE_BUILD_BENCH=ON", "-DHYLE_BUILD_TESTS=OFF",
            "-DHYLE_TEST_INSTRUMENTATION=OFF", "-DHYLE_SANITIZER=off",
            "-DHYLE_TRACE=OFF", "-DHYLE_STATIC=ON",
            "-DHYLE_BENCH_BOOST=ON", f"-DHYLE_BOOST_DIR={boost_dir}",
        ]
        if host_system != "Windows":
            configure.append(f"-DCMAKE_BUILD_TYPE={args.config}")
        if args.generator:
            configure.extend(("-G", args.generator))
        if args.architecture:
            configure.extend(("-A", args.architecture))
        commands.append(configure)
        run_and_capture(configure, os.environ.copy(),
                        output_dir / "configure.log")
    if not args.skip_build:
        targets = [
            "hyle_container_intrusive_bench",
            "hyle_container_intrusive_instrument_bench",
        ]
        if args.placement:
            targets.append("hyle_container_intrusive_bench_placement")
        build_command = [
            "cmake", "--build", str(build), "--config", args.config,
            "--target", *targets, "--parallel", "2",
        ]
        commands.append(build_command)
        run_and_capture(
            build_command, os.environ.copy(), output_dir / "build.log")

    try:
        primary = find_executable(
            build, args.config, "hyle_container_intrusive_bench")
        instrument = find_executable(
            build, args.config,
            "hyle_container_intrusive_instrument_bench")
        placement_executable = None
        if args.placement:
            placement_executable = find_executable(
                build, args.config,
                "hyle_container_intrusive_bench_placement")
    except (RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))

    placement_record = None
    lifetime_codegen_record = {}
    artifact_architecture_record = None
    tool = None
    primary_object = None
    instrument_object = None
    placement_object = None
    if args.llvm_objdump:
        tool = pathlib.Path(args.llvm_objdump).resolve()
        if not tool.is_file():
            parser.error(f"llvm-objdump unavailable: {tool}")
        try:
            primary_object = find_benchmark_object(
                build, args.config, "hyle_container_intrusive_bench",
                "intrusive_list_bench")
            instrument_object = find_benchmark_object(
                build, args.config,
                "hyle_container_intrusive_instrument_bench",
                "instrument_bench")
        except (RuntimeError, FileNotFoundError) as error:
            parser.error(str(error))
    if args.placement:
        try:
            placement_object = find_benchmark_object(
                build, args.config,
                "hyle_container_intrusive_bench_placement",
                "intrusive_list_bench")
        except (RuntimeError, FileNotFoundError) as error:
            parser.error(str(error))

    # Bind every executable, benchmark object, audit tool, and build-contract
    # file before any structural audit reads them.  The identical binding
    # taken after the audits closes the audit-to-sampling substitution window.
    try:
        audit_binding_before = artifact_binding(
            primary, instrument, placement_executable, primary_object,
            instrument_object, placement_object, build, tool)
    except (RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))

    try:
        cmake_contract = cmake_cache_values(build)
        validate_cached_peer_path(cmake_contract, boost_dir)
        compiler = compiler_record(build)
        if args.require_clean:
            validate_formal_compiler(
                host_system, compiler, build_environment)
            validate_formal_generator(host_system, cmake_contract)
    except (RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))

    try:
        filesystem_contract = filesystem_contract_record(
            host_system, (("source", repo), ("build", build),
                          ("output", output_dir), ("peer", boost_dir)),
            args.require_clean)
        no_ipo_record = verify_no_interprocedural_optimization(
            build, output_dir)
        release_optimization_record = verify_release_optimization(
            build, host_system, args.placement, output_dir)
        if tool is not None:
            lifetime_codegen_record["primary"] = lifetime_workload_codegen(
                tool, primary_object, output_dir, "primary")
            if placement_object is not None:
                lifetime_codegen_record["placement"] = \
                    lifetime_workload_codegen(
                        tool, placement_object, output_dir, "placement")
        if args.placement:
            placement_record = verify_placement(
                host_system, tool, build, primary, placement_executable,
                primary_object, placement_object, output_dir)
    except (RuntimeError, FileNotFoundError, ET.ParseError) as error:
        parser.error(str(error))

    if tool is not None:
        architecture_artifacts = [
            ("primary executable", primary),
            ("instrument executable", instrument),
            ("primary benchmark object", primary_object),
            ("instrument benchmark object", instrument_object),
        ]
        if placement_executable is not None:
            architecture_artifacts.extend((
                ("placement executable", placement_executable),
                ("placement benchmark object", placement_object),
            ))
        try:
            artifact_architecture_record = verify_artifact_architecture(
                tool, host_system, architecture_artifacts, output_dir)
        except RuntimeError as error:
            parser.error(str(error))

    try:
        artifacts_before_sampling = artifact_binding(
            primary, instrument, placement_executable, primary_object,
            instrument_object, placement_object, build, tool)
    except (RuntimeError, FileNotFoundError) as error:
        parser.error(str(error))
    if artifacts_before_sampling != audit_binding_before:
        parser.error(
            "benchmark artifacts/build binding changed during structural "
            "audits")

    try:
        execution_controls = apply_execution_controls(
            args.affinity_mask, args.priority_class)
        host_contract_before_sampling = stable_host_contract_snapshot(
            host_system, args.affinity_mask)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if args.prior_inconclusive_manifest is not None:
        try:
            prior_inconclusive_record = validate_prior_inconclusive_manifest(
                args.prior_inconclusive_manifest, host_system, source_git,
                peer_git, input_hashes_before, host_identity, compiler,
                cmake_contract, build_environment, execution_controls,
                host_contract_before_sampling, args.apple_qos)
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            parser.error(str(error))

    grouped_shapes = {}
    logs = []
    product_shapes = [("primary", primary)]
    if placement_executable is not None:
        product_shapes.append(("placement", placement_executable))
    for shape, executable in product_shapes:
        grouped_shapes[shape] = {}
        for allocation, peer_allocation_first in ALLOCATION_STRATA:
            grouped_shapes[shape][allocation] = {}
            for peer_first in (False, True):
                order = "peer-first" if peer_first else "candidate-first"
                command = bench_stat_command(
                    bench_stat_path, executable, args.runs,
                    args.equivalence_margin_pct, boost_dir,
                    PRODUCT_PAIRS, [], args.require_clean,
                    PRODUCT_TELEMETRY_PREFIX)
                environment = os.environ.copy()
                environment[
                    "HYLE_CONTAINER_INTRUSIVE_BENCH_FORMAL_ONLY"] = "1"
                if args.apple_qos is not None:
                    environment["HYLE_BENCH_APPLE_QOS"] = args.apple_qos
                if peer_first:
                    environment["HYLE_BENCH_RIVAL_FIRST"] = "1"
                else:
                    environment.pop("HYLE_BENCH_RIVAL_FIRST", None)
                if peer_allocation_first:
                    environment[
                        "HYLE_BENCH_PEER_ALLOCATION_FIRST"] = "1"
                else:
                    environment.pop(
                        "HYLE_BENCH_PEER_ALLOCATION_FIRST", None)
                path = output_dir / (
                    f"product-{shape}-{allocation}-{order}.log")
                commands.append(command)
                text = run_and_capture(command, environment, path)
                qos_name = "not-applicable"
                qos_ok = "not-applicable"
                if host_system == "Darwin":
                    qos_name = ("user-interactive" if args.apple_qos !=
                                "utility" else "utility")
                    qos_ok = "1"
                product_marker = (
                    PRODUCT_TELEMETRY_PREFIX +
                    f"formal_only=1 execution_order={order} "
                    f"allocation_order={allocation} "
                    f"apple_qos={qos_name} qos_ok={qos_ok} "
                    "affinity=external")
                grouped_shapes[shape][allocation][order] = validate_log(
                    text, path, args.runs, PRODUCT_CONTRACTS,
                    (PRODUCT_TELEMETRY_PREFIX, product_marker))
                logs.append({
                    "kind": "product", "shape": shape,
                    "allocation_order": allocation,
                    "execution_order": order,
                    "benchmark_environment": {
                        "HYLE_CONTAINER_INTRUSIVE_BENCH_FORMAL_ONLY": "1",
                        "HYLE_BENCH_RIVAL_FIRST": (
                            "1" if peer_first else None),
                        "HYLE_BENCH_PEER_ALLOCATION_FIRST": (
                            "1" if peer_allocation_first else None),
                        "HYLE_BENCH_APPLE_QOS": args.apple_qos,
                    },
                    "path": str(path), "sha256": sha256(path),
                    "raw_pair_records": args.runs * len(PRODUCT_PAIRS),
                })

    grouped_controls = {}
    for peer_first in (False, True):
        order = "peer-first" if peer_first else "candidate-first"
        command = bench_stat_command(
            bench_stat_path, instrument, args.runs,
            args.equivalence_margin_pct, boost_dir,
            [AA_PAIR, *DELTA_PAIRS],
            [TIMER_FLOOR_PAIR, PHASE_FLOOR_PAIR], args.require_clean,
            INSTRUMENT_TELEMETRY_PREFIX)
        environment = os.environ.copy()
        if args.apple_qos is not None:
            environment["HYLE_BENCH_APPLE_QOS"] = args.apple_qos
        if peer_first:
            environment["HYLE_BENCH_RIVAL_FIRST"] = "1"
        else:
            environment.pop("HYLE_BENCH_RIVAL_FIRST", None)
        path = output_dir / f"instrument-{order}.log"
        commands.append(command)
        text = run_and_capture(command, environment, path)
        qos_name = "not-applicable"
        qos_ok = "not-applicable"
        if host_system == "Darwin":
            qos_name = ("user-interactive" if args.apple_qos !=
                        "utility" else "utility")
            qos_ok = "1"
        instrument_marker = (
            INSTRUMENT_TELEMETRY_PREFIX +
            f"execution_order={order} apple_qos={qos_name} "
            f"qos_ok={qos_ok} affinity=external")
        grouped_controls[order] = validate_log(
            text, path, args.runs, CONTROL_CONTRACTS,
            (INSTRUMENT_TELEMETRY_PREFIX, instrument_marker))
        logs.append({
            "kind": "instrument", "shape": "separate-control-binary",
            "execution_order": order,
            "path": str(path), "sha256": sha256(path),
            "benchmark_environment": {
                "HYLE_BENCH_RIVAL_FIRST": (
                    "1" if peer_first else None),
                "HYLE_BENCH_APPLE_QOS": args.apple_qos,
            },
            "raw_pair_records": args.runs * len(CONTROL_CONTRACTS),
        })

    host_contract_after_sampling = stable_host_contract_snapshot(
        host_system, args.affinity_mask)
    if host_contract_after_sampling != host_contract_before_sampling:
        raise RuntimeError(
            "host affinity/power/topology contract changed during sampling")
    if collect_input_hashes(repo) != input_hashes_before:
        raise RuntimeError("benchmark/product inputs changed during sampling")
    artifacts_after_sampling = artifact_binding(
        primary, instrument, placement_executable, primary_object,
        instrument_object, placement_object, build, tool)
    if artifacts_after_sampling != artifacts_before_sampling:
        raise RuntimeError("benchmark artifacts/build binding changed during sampling")
    source_git_after = git_record(repo)
    peer_git_after = git_record(boost_dir)
    if source_git_after != source_git or peer_git_after != peer_git:
        raise RuntimeError("source or peer checkout changed during sampling")
    formal_artifact_archive = None
    artifacts_after_archive = None
    if args.require_clean:
        formal_artifact_archive = archive_formal_artifacts(
            output_dir, artifacts_after_sampling, build)
        artifacts_after_archive = artifact_binding(
            primary, instrument, placement_executable, primary_object,
            instrument_object, placement_object, build, tool)
        if artifacts_after_archive != artifacts_after_sampling:
            raise RuntimeError(
                "benchmark artifacts/build binding changed during formal "
                "archive")

    bench_stat = load_bench_stat(bench_stat_path)
    product_summary = summarize_product(
        bench_stat, grouped_shapes, args.equivalence_margin_pct,
        placement_executable is not None)
    controls_summary = summarize_controls(
        bench_stat, grouped_controls, grouped_shapes,
        args.equivalence_margin_pct)
    product_gates = [
        row["platform_sota_noninferiority_gate"]
        for row in product_summary
    ]
    structural_gate = (
        "PASS" if no_ipo_record["gate"] == "PASS" and
        release_optimization_record["gate"] == "PASS" and
        artifact_architecture_record is not None and
        artifact_architecture_record["gate"] == "PASS" and
        set(lifetime_codegen_record) == (
            {"primary", "placement"} if placement_executable is not None
            else {"primary"}) and
        all(record["gate"] == "PASS"
            for record in lifetime_codegen_record.values()) else
        "UNRESOLVED")
    if (controls_summary["instrument_gate"] != "PASS" or
            structural_gate != "PASS"):
        platform_gate = "INCONCLUSIVE"
    elif any(gate == "FAIL" for gate in product_gates):
        platform_gate = "FAIL"
    elif all(gate == "PASS" for gate in product_gates):
        platform_gate = "PASS"
    else:
        platform_gate = "INCONCLUSIVE"
    summary = {
        "record": "hyle_container_intrusive_platform_summary",
        "schema": 3,
        "platform": host_system,
        "placement_evaluated": placement_executable is not None,
        "ratio_definition": "RIVAL op/s / HYLE BASELINE op/s",
        "equivalence_margin_pct": args.equivalence_margin_pct,
        "controls": controls_summary,
        "structural_workload_gate": structural_gate,
        "product_pairs": product_summary,
        "platform_instrument_and_noninferiority_gate": platform_gate,
        "cross_platform_gate_complete": False,
    }
    summary_path = output_dir / "platform-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    input_hashes_after = collect_input_hashes(repo)
    build_logs = []
    for name in ("configure.log", "build.log"):
        path = output_dir / name
        if path.is_file():
            build_logs.append({
                "path": str(path), "sha256": sha256(path)})
    manifest = {
        "record": "hyle_container_intrusive_bench_evidence_manifest",
        "schema": 2,
        "observed_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "qualification": (
            "formal-clean-" + host_system.lower() if args.require_clean
            else "diagnostic-dirty-integration"),
        "host": host_identity,
        "runs_per_order": args.runs,
        "equivalence_margin_pct": args.equivalence_margin_pct,
        "source_git_at_start": source_git,
        "source_git_at_end": source_git_after,
        "peer_git_at_start": peer_git,
        "peer_git_at_end": peer_git_after,
        "input_files_sha256": input_hashes_before,
        "input_hashes_stable_during_run": (
            input_hashes_before == input_hashes_after),
        "cmake_contract": cmake_contract,
        "compiler": compiler,
        "build_environment": build_environment,
        "filesystem_contract": filesystem_contract,
        "audit_binding_before": audit_binding_before,
        "artifacts_before_sampling": artifacts_before_sampling,
        "audit_binding_stable": (
            audit_binding_before == artifacts_before_sampling),
        "artifacts_after_sampling": artifacts_after_sampling,
        "artifacts_stable_during_sampling": (
            artifacts_before_sampling == artifacts_after_sampling),
        "artifacts_after_archive": artifacts_after_archive,
        "artifacts_stable_during_archive": (
            None if artifacts_after_archive is None else
            artifacts_after_sampling == artifacts_after_archive),
        "formal_artifact_archive": formal_artifact_archive,
        "configure_and_build_logs": build_logs,
        "cmake_version": command_text(["cmake", "--version"]).splitlines()[0],
        "python_version": platform.python_version(),
        "execution_controls": execution_controls,
        "host_contract_before_sampling": host_contract_before_sampling,
        "host_contract_after_sampling": host_contract_after_sampling,
        "host_contract_stable_during_sampling": (
            host_contract_before_sampling == host_contract_after_sampling),
        "host_telemetry": host_telemetry(host_system, args.affinity_mask),
        "apple_qos_requested": args.apple_qos,
        "runtime_qos_telemetry_validated": True,
        "runtime_strata_telemetry_validated": True,
        "prior_inconclusive_n15": prior_inconclusive_record,
        "active_power_scheme": host_contract_before_sampling.get(
            "active_power_scheme"),
        "linkage_contract": {
            "hyle": "static/header-only", "boost": "header-only",
            "same_final_executable": True,
        },
        "storage_address_strata": {
            "candidate_allocation_first":
                "HYLE allocation first, Boost second",
            "peer_allocation_first":
                "Boost allocation first, HYLE second",
            "execution_order": "independently candidate-first/peer-first",
            "gate": "complete 2x2 allocation/execution strata required",
            "heap_preheat": (
                "not a separate dimension: node allocation is outside the "
                "timed contract; reversed live allocation order is the "
                "caller-storage address perturbation"),
        },
        "address_space_layout_randomization": {
            "mode": "host default", "toggled_during_experiment": False,
        },
        "executables": artifacts_after_sampling["executables"],
        "placement": placement_record,
        "artifact_architecture": artifact_architecture_record,
        "no_interprocedural_optimization": no_ipo_record,
        "release_optimization": release_optimization_record,
        "lifetime_workload_codegen": lifetime_codegen_record,
        "commands": commands,
        "logs": logs,
        "summary": {"path": str(summary_path),
                    "sha256": sha256(summary_path)},
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"evidence manifest: {manifest_path}", flush=True)
    print(f"{host_system} gate: {platform_gate}", flush=True)
    if controls_summary["instrument_gate"] != "PASS":
        raise RuntimeError("instrument qualification gate failed")
    if args.require_clean and platform_gate != "PASS":
        raise RuntimeError(
            f"formal {host_system} non-inferiority gate did not pass: " +
            platform_gate)


if __name__ == "__main__":
    main()
