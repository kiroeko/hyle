#!/usr/bin/env python3
"""Run only the error benchmarks and archive fail-closed paired evidence."""

import argparse
import ctypes
import datetime
import gzip
import hashlib
import importlib.util
import json
import locale
import os
import pathlib
import platform
import re
import shlex
import subprocess
import sys


RAW_PREFIX = "# HYLE_BENCH_PAIR_RAW "

STATUS_PAIRS = [
    ("HYLE Status::Ok", "absl::OkStatus"),
    ("HYLE Status copy (OK)", "absl::Status copy (OK)"),
    (
        "HYLE Status runtime code, no message (carrier escapes)",
        "absl::Status runtime code, empty, no-loc (carrier escapes)",
    ),
    (
        "HYLE Status::Cancelled() + code() (carrier escapes)",
        "absl::CancelledError() + code() (carrier escapes)",
    ),
    (
        "HYLE Result OK (4 noinline frames + unwrap)",
        "absl::StatusOr OK (4 noinline frames + unwrap)",
    ),
    (
        "HYLE Result code-only error (4 noinline frames)",
        "absl::StatusOr code-only error (4 noinline frames)",
    ),
]

NORMAL_PAIRS = [
    (
        "HYLE Status::NotFound owning 7-byte SSO-sized message "
        "(borrowed input)",
        "absl::Status owning 7-byte SSO-sized message "
        "(borrowed input, no-loc)",
    ),
    (
        "HYLE Status::NotFound owning 29-byte message (borrowed input)",
        "absl::Status owning 29-byte message (borrowed input, no-loc)",
    ),
    (
        "HYLE Status::NotFound owning 256-byte long message (borrowed input)",
        "absl::Status owning 256-byte long message "
        "(borrowed input, no-loc)",
    ),
    (
        "HYLE Status copy (shared owned 29-byte message)",
        "absl::Status copy (shared owned 29-byte message)",
    ),
    (
        "HYLE Status::WithContext x1 (copied lvalue seed)",
        "absl::StatusBuilder SetPrepend x1 (prebuilt lvalue seed)",
    ),
    (
        "HYLE Status::WithContext x3 (copied lvalue seed, rvalue chain)",
        "absl::StatusBuilder SetPrepend x3 (prebuilt lvalue seed)",
    ),
    (
        "HYLE Status::WithContext x16 (fresh seed, rvalue chain)",
        "absl::StatusBuilder SetPrepend x16 (fresh seed, rvalue chain)",
    ),
    (
        "HYLE Status::WithContext x16 (prebuilt seed, lvalue each layer)",
        "absl::StatusBuilder SetPrepend x16 "
        "(prebuilt seed, lvalue each layer)",
    ),
    (
        "HYLE Result message error (2 noinline frames)",
        "absl::StatusOr message error (2 noinline frames)",
    ),
    (
        "HYLE hyle::to_string owning (54-byte message)",
        "absl::Status::ToString owning (54-byte message)",
    ),
    (
        "HYLE hyle::to_string owning (512-byte message)",
        "absl::Status::ToString owning (512-byte message)",
    ),
    (
        "HYLE Status::Update lvalue message errors x2",
        "absl::Status::Update lvalue message errors x2",
    ),
    (
        "HYLE Status::Update noinline factory prvalues x2",
        "absl::Status::Update noinline factory prvalues x2",
    ),
]

NORMAL_NOT_COMPARABLE = [
    ("HYLE status_from_errno", "absl::ErrnoToStatus + native payload"),
    ("HYLE Status::FromNative", "absl::Status + SetPayload"),
    ("HYLE Status::native_error", "absl::Status GetPayload + decode"),
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
            f"command exited {result.returncode}: {display_command(command)}\n"
            f"{result.stdout}{result.stderr}")
    return (result.stdout + result.stderr).strip()


def git_record(directory):
    status = command_text(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        directory)
    return {
        "head": command_text(["git", "rev-parse", "HEAD"], directory),
        "clean": not bool(status),
        "status_porcelain": status.splitlines(),
        "status_sha256": hashlib.sha256(
            status.encode("utf-8")).hexdigest(),
    }


def collect_input_hashes(repo):
    patterns = [
        "CMakeLists.txt",
        "cmake/**/*.cmake",
        "bench/CMakeLists.txt",
        "bench/common/**/*",
        "bench/error/**/*",
        "bench/peers/CMakeLists.txt",
        "include/hyle/error/**/*",
        "src/error/**/*",
        "scripts/bench-stat.py",
        "scripts/run-error-bench-evidence.py",
    ]
    paths = set()
    for pattern in patterns:
        paths.update(path for path in repo.glob(pattern) if path.is_file())
    return {
        path.relative_to(repo).as_posix(): sha256(path)
        for path in sorted(paths)
    }


def collect_build_bindings(build):
    paths = []
    cache = build / "CMakeCache.txt"
    if cache.is_file():
        paths.append(cache)
    paths.extend(build.rglob("hyle_error_bench*.vcxproj"))
    paths.extend(build.rglob("hyle_error_normal_bench*.vcxproj"))
    return {
        path.relative_to(build).as_posix(): sha256(path)
        for path in sorted(set(paths))
    }


def verify_cmake_contract(build):
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
        "HYLE_COMPONENTS": "error",
        "HYLE_BUILD_BENCH": "ON",
        "HYLE_BUILD_TESTS": "OFF",
        "HYLE_TEST_INSTRUMENTATION": "OFF",
        "HYLE_STATIC": "ON",
        "HYLE_BENCH_ABSL": "ON",
    }
    mismatches = {
        key: {"expected": expected_value, "observed": values.get(key)}
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(
            "error benchmark CMake contract mismatch: " +
            json.dumps(mismatches, sort_keys=True))
    return {"cache": str(cache), "sha256": sha256(cache), "values": values}


def compiler_record(build):
    candidates = list(build.glob(
        "CMakeFiles/*/CMakeCXXCompiler.cmake"))
    if not candidates:
        return None
    path = candidates[0]
    wanted = {
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_COMPILER_ID",
        "CMAKE_CXX_COMPILER_VERSION",
        "CMAKE_CXX_COMPILER_ARCHITECTURE_ID",
        "CMAKE_CXX_SIMULATE_ID",
        "CMAKE_CXX_SIMULATE_VERSION",
    }
    values = {}
    for line in path.read_text(
            encoding="utf-8", errors="replace").splitlines():
        match = re.fullmatch(r'set\(([^ ]+) "(.*)"\)', line)
        if match and match.group(1) in wanted:
            values[match.group(1)] = match.group(2)
    return {
        "path": str(path),
        "sha256": sha256(path),
        "values": values,
    }


def windows_kernel32():
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    kernel32.GetProcessAffinityMask.restype = ctypes.c_int
    kernel32.SetProcessAffinityMask.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    kernel32.SetProcessAffinityMask.restype = ctypes.c_int
    kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
    kernel32.GetPriorityClass.restype = ctypes.c_uint32
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    return kernel32


def execution_environment_record(requested_affinity, requested_priority):
    record = {
        "logical_cpu_count": os.cpu_count(),
        "benchmark_processes_sequential": True,
        "requested_affinity_mask": requested_affinity,
        "requested_priority_class": requested_priority,
        "background_load_control": "not controlled by runner",
    }
    if os.name == "nt":
        power = command_text(
            ["powercfg", "/GETACTIVESCHEME"],
            encoding=locale.getpreferredencoding(False))
        record["windows_active_power_scheme"] = power
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        kernel32 = windows_kernel32()
        handle = kernel32.GetCurrentProcess()
        if kernel32.GetProcessAffinityMask(
                handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            record["inherited_process_affinity_mask"] = hex(process_mask.value)
            record["system_affinity_mask"] = hex(system_mask.value)
        record["windows_priority_class"] = hex(
            kernel32.GetPriorityClass(handle))
    elif hasattr(os, "sched_getaffinity"):
        record["inherited_cpu_affinity"] = sorted(os.sched_getaffinity(0))
    return record


def apply_benchmark_execution_controls(affinity_text, priority):
    affinity = None if affinity_text is None else int(affinity_text, 0)
    if affinity is not None and affinity <= 0:
        raise ValueError("affinity mask must select at least one CPU")
    if os.name == "nt":
        kernel32 = windows_kernel32()
        handle = kernel32.GetCurrentProcess()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
                handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise RuntimeError("GetProcessAffinityMask failed")
        if affinity is not None:
            if affinity & ~system_mask.value:
                raise ValueError(
                    f"affinity {hex(affinity)} exceeds available mask "
                    f"{hex(system_mask.value)}")
            if not kernel32.SetProcessAffinityMask(
                    handle, ctypes.c_size_t(affinity)):
                raise RuntimeError("SetProcessAffinityMask failed")
        priority_values = {
            "normal": 0x00000020,
            "above-normal": 0x00008000,
            "high": 0x00000080,
        }
        if not kernel32.SetPriorityClass(handle, priority_values[priority]):
            raise RuntimeError("SetPriorityClass failed")
    else:
        if affinity is not None:
            if not hasattr(os, "sched_setaffinity"):
                raise RuntimeError("CPU affinity control is unavailable")
            cpus = {
                bit for bit in range(affinity.bit_length())
                if affinity & (1 << bit)
            }
            os.sched_setaffinity(0, cpus)
        if priority != "normal":
            raise RuntimeError(
                "non-Windows priority control is not implemented")


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
    raise FileNotFoundError(
        f"cannot find {stem + suffix} under build tree {build}")


def find_benchmark_object(build, config, target, source_stem):
    suffix = ".obj" if os.name == "nt" else ".o"
    candidates = [
        path for path in build.rglob(source_stem + suffix)
        if any(part.lower() == target.lower() + ".dir" for part in path.parts)
    ]
    configured = [
        path for path in candidates
        if config.lower() in [part.lower() for part in path.parts]
    ]
    candidates = configured or candidates
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"expected one {target} {source_stem}{suffix} under {build}; "
            f"found {len(candidates)}")
    return candidates[0].resolve()


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


def bench_stat_command(bench_stat, executable, runs, margin, absl_dir,
                       comparable_pairs, not_comparable_pairs, require_clean):
    command = [
        sys.executable,
        str(bench_stat),
        str(executable),
        "-n",
        str(runs),
        "--peer",
        f"abseil={absl_dir}",
        "--equivalence-margin-pct",
        str(margin),
    ]
    if require_clean:
        command.append("--require-clean")
    for baseline, rival in comparable_pairs:
        command.extend(("--pair", baseline, rival))
    for baseline, rival in not_comparable_pairs:
        command.extend(("--not-comparable-pair", baseline, rival))
    return command


def validate_log(text, path, runs, pair_count):
    if "(no rows matched pair:" in text:
        raise RuntimeError(f"pair manifest did not match benchmark rows: {path}")
    raw_records = [
        json.loads(line[len(RAW_PREFIX):])
        for line in text.splitlines() if line.startswith(RAW_PREFIX)
    ]
    raw_count = len(raw_records)
    expected = runs * pair_count
    if raw_count != expected:
        raise RuntimeError(
            f"raw record count mismatch in {path}: got {raw_count}, "
            f"expected {expected}")
    return raw_count, raw_records


def validate_pair_records(records, path, runs, comparable_pairs,
                          not_comparable_pairs):
    expected = {
        pair: True for pair in comparable_pairs
    } | {
        pair: False for pair in not_comparable_pairs
    }
    grouped = {}
    for record in records:
        pair = (record.get("baseline"), record.get("rival"))
        if pair not in expected:
            raise RuntimeError(f"unexpected raw pair in {path}: {pair}")
        if record.get("comparable") is not expected[pair]:
            raise RuntimeError(f"comparability mismatch in {path}: {pair}")
        grouped.setdefault(pair, []).append(record)
    if set(grouped) != set(expected):
        missing = sorted(set(expected) - set(grouped))
        raise RuntimeError(f"raw pair manifest incomplete in {path}: {missing}")
    expected_runs = list(range(1, runs + 1))
    for pair, pair_records in grouped.items():
        observed_runs = sorted(record.get("run") for record in pair_records)
        if observed_runs != expected_runs:
            raise RuntimeError(
                f"raw run ids mismatch in {path} for {pair}: "
                f"{observed_runs}")
    return grouped


def load_bench_stat(path):
    spec = importlib.util.spec_from_file_location("hyle_bench_stat", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load statistical runner: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def order_summary(bench_stat, records, margin, comparable):
    ratios = [record["ratio_rival_over_baseline"] for record in records]
    decision = bench_stat.paired_verdict(
        ratios, margin, comparable=comparable)
    gate = bench_stat.sota_noninferiority_gate(
        ratios, margin, comparable=comparable)
    interval = decision["interval"]
    return {
        "ratio_median": decision["median"],
        "median_confidence_interval": None if interval is None else interval[:2],
        "confidence_order_statistic_k": None if interval is None else interval[2],
        "exact_verdict": decision["verdict"],
        "exact_reason": decision["reason"],
        "sota_gate": gate["gate"],
        "sota_reason": gate["reason"],
    }


def summarize_one_shape(bench_stat, grouped_orders, benchmark, shape,
                        pair, margin, comparable):
    first = order_summary(
        bench_stat,
        grouped_orders[(benchmark, shape, "candidate-first")][pair],
        margin, comparable)
    second = order_summary(
        bench_stat,
        grouped_orders[(benchmark, shape, "peer-first")][pair],
        margin, comparable)
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


def summarize_dual_orders(bench_stat, grouped_orders, margin,
                          placement_available):
    records = []
    manifests = [
        ("status", STATUS_PAIRS, []),
        ("normal", NORMAL_PAIRS, NORMAL_NOT_COMPARABLE),
    ]
    for benchmark, comparable_pairs, not_comparable_pairs in manifests:
        for pair, comparable in (
                [(pair, True) for pair in comparable_pairs] +
                [(pair, False) for pair in not_comparable_pairs]):
            primary = summarize_one_shape(
                bench_stat, grouped_orders, benchmark, "primary", pair,
                margin, comparable)
            placement = None
            if placement_available:
                placement = summarize_one_shape(
                    bench_stat, grouped_orders, benchmark, "placement", pair,
                    margin, comparable)
                formal_verdict = bench_stat.combine_order_verdicts(
                    primary["dual_order_exact_verdict"],
                    placement["dual_order_exact_verdict"],
                    comparable=comparable, shape_stable=True)
                formal_gate = bench_stat.combine_sota_gates(
                    primary["dual_order_sota_gate"],
                    placement["dual_order_sota_gate"],
                    comparable=comparable)
            else:
                formal_verdict = bench_stat.combine_order_verdicts(
                    primary["dual_order_exact_verdict"],
                    primary["dual_order_exact_verdict"],
                    comparable=comparable, shape_stable=not comparable)
                formal_gate = bench_stat.combine_sota_gates(
                    primary["dual_order_sota_gate"],
                    primary["dual_order_sota_gate"],
                    shape="UNRESOLVED" if comparable else "N/A",
                    comparable=comparable)
            records.append({
                "benchmark": benchmark,
                "baseline": pair[0],
                "rival": pair[1],
                "comparable": comparable,
                "primary": primary,
                "placement": placement,
                "formal_exact_verdict_with_current_gates": formal_verdict,
                "formal_sota_gate_with_current_gates": formal_gate,
            })
    return records


def function_segment(disassembly, name):
    marker = f"::{name}(int)>:"
    marker_at = disassembly.find(marker)
    if marker_at < 0:
        return ""
    start = disassembly.rfind("\n", 0, marker_at) + 1
    match = re.search(r"\n[0-9A-Fa-f]+ <", disassembly[marker_at:])
    if match is None:
        return disassembly[start:]
    return disassembly[start:marker_at + match.start()]


def verify_code_shape(tool, status_object, output_dir, label):
    command = [
        str(tool), "--demangle", "--reloc", "--disassemble",
        str(status_object),
    ]
    disassembly = command_text(command)
    raw_path = output_dir / f"status-{label}-code-shape.objdump.txt.gz"
    with gzip.open(raw_path, "wt", encoding="utf-8", newline="\n") as raw:
        raw.write(disassembly)
        raw.write("\n")

    edges = [
        ("hyle_chain_error", "hyle_error_layer2"),
        ("hyle_error_layer2", "hyle_error_layer1"),
        ("hyle_error_layer1", "hyle_leaf_error"),
        ("std_chain_error", "std_error_layer2"),
        ("std_error_layer2", "std_error_layer1"),
        ("std_error_layer1", "std_leaf_error"),
        ("absl_chain_error", "absl_error_layer2"),
        ("absl_error_layer2", "absl_error_layer1"),
        ("absl_error_layer1", "absl_leaf_error"),
    ]
    checks = []
    for caller, callee in edges:
        segment = function_segment(disassembly, caller)
        passed = bool(segment) and callee in segment and "IMAGE_REL" in segment
        checks.append({"caller": caller, "callee": callee, "passed": passed})
    opaque_relocations = len(re.findall(
        r"IMAGE_REL[^\n]*opaque_code_choice", disassembly))
    barrier_relocations = len(re.findall(
        r"IMAGE_REL[^\n]*doNotOptimizeAwaySink", disassembly))
    if not all(check["passed"] for check in checks):
        raise RuntimeError("four-frame code-shape call-edge check failed")
    if opaque_relocations < 3 or barrier_relocations < 3:
        raise RuntimeError(
            "runtime-code selector/barrier relocations were optimized away")

    record = {
        "tool": str(tool),
        "tool_version": command_text([str(tool), "--version"]).splitlines()[0],
        "command": command,
        "object": str(status_object),
        "object_sha256": sha256(status_object),
        "raw_gzip": str(raw_path),
        "raw_gzip_sha256": sha256(raw_path),
        "four_frame_edges": checks,
        "opaque_code_choice_relocations": opaque_relocations,
        "do_not_optimize_relocations": barrier_relocations,
        "shape_label": label,
    }
    summary_path = output_dir / f"status-{label}-code-shape.json"
    summary_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return record


def pe_text_section(tool, executable):
    command = [str(tool), "--section-headers", str(executable)]
    output = command_text(command)
    match = re.search(
        r"^\s*\d+\s+\.text\s+([0-9A-Fa-f]+)\s+([0-9A-Fa-f]+)\s+TEXT\s*$",
        output, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"cannot read PE .text section: {executable}")
    return {
        "executable": str(executable),
        "sha256": sha256(executable),
        "text_size_bytes": int(match.group(1), 16),
        "text_vma": int(match.group(2), 16),
        "command": command,
    }


def normalized_object_code(tool, path):
    command = [str(tool), "--disassemble", "--reloc", str(path)]
    lines = command_text(command).splitlines()
    if lines and "file format" in lines[0]:
        lines[0] = "file format" + lines[0].split("file format", 1)[1]
    text = "\n".join(lines) + "\n"
    return {
        "path": str(path),
        "object_sha256": sha256(path),
        "normalized_disassembly_sha256": hashlib.sha256(
            text.encode("utf-8")).hexdigest(),
        "command": command,
    }


def verify_placement_layout(tool, primary_executables, placement_executables,
                            primary_objects, placement_objects, output_dir):
    records = {}
    for name in ("status", "normal"):
        primary = pe_text_section(tool, primary_executables[name])
        placement = pe_text_section(tool, placement_executables[name])
        if placement["text_size_bytes"] <= primary["text_size_bytes"]:
            raise RuntimeError(
                f"/FUNCTIONPADMIN did not increase {name} .text size")
        primary_object = normalized_object_code(tool, primary_objects[name])
        placement_object = normalized_object_code(
            tool, placement_objects[name])
        if (primary_object["normalized_disassembly_sha256"] !=
                placement_object["normalized_disassembly_sha256"]):
            raise RuntimeError(
                f"primary/placement compiler output differs for {name}")
        records[name] = {
            "primary": primary,
            "placement": placement,
            "text_size_delta_bytes": (
                placement["text_size_bytes"] - primary["text_size_bytes"]),
            "primary_object": primary_object,
            "placement_object": placement_object,
            "object_code_identical_before_link_padding": True,
        }
    record = {
        "mechanism": "MSVC LINK /FUNCTIONPADMIN:64",
        "contract_effect": "inter-function padding only",
        "candidate_and_peer_share_executable": True,
        "records": records,
    }
    path = output_dir / "placement-layout.json"
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    record["path"] = str(path)
    record["sha256"] = sha256(path)
    return record


def main():
    parser = argparse.ArgumentParser(
        description="fresh-build and run only error paired evidence")
    parser.add_argument("--build", required=True,
                        help="configured CMake build tree")
    parser.add_argument("--absl-dir", required=True,
                        help="clean pinned abseil-cpp checkout")
    parser.add_argument("--output-dir", required=True,
                        help="new directory for build/raw logs and manifest")
    parser.add_argument("--config", default="Release",
                        help="multi-config build configuration (default Release)")
    parser.add_argument(
        "--fresh-configure", action="store_true",
        help="require a new build tree and configure only error benchmarks")
    parser.add_argument(
        "--generator", help="CMake generator used with --fresh-configure")
    parser.add_argument(
        "--architecture", help="CMake -A value used with --fresh-configure")
    parser.add_argument(
        "--llvm-objdump",
        help="llvm-objdump path for fail-closed propagation/code shape checks")
    parser.add_argument(
        "--placement", action="store_true",
        help="build and sample the MSVC /FUNCTIONPADMIN placement shape")
    parser.add_argument(
        "--affinity-mask",
        help="CPU mask inherited by bench-stat and benchmark child processes")
    parser.add_argument(
        "--priority-class", default="normal",
        choices=("normal", "above-normal", "high"),
        help="Windows priority inherited by benchmark children")
    parser.add_argument("-n", "--runs", type=int, default=15,
                        help="independent processes per order (default 15)")
    parser.add_argument("--equivalence-margin-pct", type=float, default=3.0,
                        help="predeclared margin shared by error hot pairs")
    parser.add_argument("--require-clean", action="store_true",
                        help="enable bench-stat clean source/peer gate")
    parser.add_argument("--skip-build", action="store_true",
                        help="use already-built executables")
    args = parser.parse_args()

    if args.runs < 15:
        parser.error("error evidence requires at least 15 independent processes")
    if args.equivalence_margin_pct <= 0.0:
        parser.error("equivalence margin must be positive")
    if args.fresh_configure and args.skip_build:
        parser.error("--fresh-configure and --skip-build are incompatible")
    if (args.generator or args.architecture) and not args.fresh_configure:
        parser.error("--generator/--architecture require --fresh-configure")
    if args.placement and not args.llvm_objdump:
        parser.error("--placement requires --llvm-objdump layout verification")
    if args.require_clean and not args.placement:
        parser.error("formal clean error evidence requires --placement")
    if args.affinity_mask is not None:
        try:
            int(args.affinity_mask, 0)
        except ValueError:
            parser.error("--affinity-mask must be an integer such as 0x4")

    repo = pathlib.Path(__file__).resolve().parents[1]
    build = pathlib.Path(args.build).resolve()
    absl_dir = pathlib.Path(args.absl_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve()
    bench_stat = repo / "scripts" / "bench-stat.py"
    if not (absl_dir / "CMakeLists.txt").is_file():
        parser.error(f"Abseil checkout is unavailable: {absl_dir}")
    source_git = git_record(repo)
    peer_git = git_record(absl_dir)
    if args.require_clean and (not source_git["clean"] or not peer_git["clean"]):
        parser.error("--require-clean requires clean source and peer checkouts")
    try:
        output_dir.relative_to(repo)
        output_inside_repo = True
    except ValueError:
        output_inside_repo = False
    if args.require_clean and output_inside_repo:
        parser.error("--require-clean output directory must be outside source tree")
    if args.fresh_configure and build.exists():
        parser.error(f"fresh build tree already exists: {build}")
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"output directory already exists: {output_dir}")

    recorded_commands = []
    input_hashes_before = collect_input_hashes(repo)
    if args.fresh_configure:
        configure_command = [
            "cmake", "-S", str(repo), "-B", str(build),
            "-DHYLE_COMPONENTS=error",
            "-DHYLE_BUILD_BENCH=ON",
            "-DHYLE_BUILD_TESTS=OFF",
            "-DHYLE_TEST_INSTRUMENTATION=OFF",
            "-DHYLE_STATIC=ON",
            "-DHYLE_BENCH_ABSL=ON",
            f"-DHYLE_ABSL_DIR={absl_dir}",
        ]
        if args.generator:
            configure_command.extend(("-G", args.generator))
        if args.architecture:
            configure_command.extend(("-A", args.architecture))
        recorded_commands.append(configure_command)
        run_and_capture(
            configure_command, os.environ.copy(),
            output_dir / "configure.log")
    if not args.skip_build:
        build_targets = ["hyle_error_bench", "hyle_error_normal_bench"]
        if args.placement:
            build_targets.extend((
                "hyle_error_bench_placement",
                "hyle_error_normal_bench_placement",
            ))
        build_command = [
            "cmake", "--build", str(build), "--config", args.config,
            "--target", *build_targets,
            "--parallel", "4",
        ]
        recorded_commands.append(build_command)
        run_and_capture(
            build_command, os.environ.copy(), output_dir / "build.log")

    try:
        cmake_contract = verify_cmake_contract(build)
    except RuntimeError as error:
        parser.error(str(error))

    try:
        primary_executables = {
            "status": find_executable(
                build, args.config, "hyle_error_bench"),
            "normal": find_executable(
                build, args.config, "hyle_error_normal_bench"),
        }
        placement_executables = None
        if args.placement:
            placement_executables = {
                "status": find_executable(
                    build, args.config, "hyle_error_bench_placement"),
                "normal": find_executable(
                    build, args.config, "hyle_error_normal_bench_placement"),
            }
    except FileNotFoundError as error:
        parser.error(str(error))

    code_shape = {}
    placement_layout = None
    if args.llvm_objdump:
        objdump = pathlib.Path(args.llvm_objdump).resolve()
        if not objdump.is_file():
            parser.error(f"llvm-objdump is unavailable: {objdump}")
        try:
            primary_objects = {
                "status": find_benchmark_object(
                    build, args.config, "hyle_error_bench", "status_bench"),
                "normal": find_benchmark_object(
                    build, args.config, "hyle_error_normal_bench",
                    "normal_bench"),
            }
        except FileNotFoundError as error:
            parser.error(str(error))
        code_shape["primary"] = verify_code_shape(
            objdump, primary_objects["status"], output_dir, "primary")
        if placement_executables is not None:
            try:
                placement_objects = {
                    "status": find_benchmark_object(
                        build, args.config, "hyle_error_bench_placement",
                        "status_bench"),
                    "normal": find_benchmark_object(
                        build, args.config,
                        "hyle_error_normal_bench_placement", "normal_bench"),
                }
            except FileNotFoundError as error:
                parser.error(str(error))
            code_shape["placement"] = verify_code_shape(
                objdump, placement_objects["status"], output_dir,
                "placement")
            placement_layout = verify_placement_layout(
                objdump, primary_executables, placement_executables,
                primary_objects, placement_objects, output_dir)

    runs = [
        ("status", "primary", primary_executables["status"],
         STATUS_PAIRS, [], False),
        ("status", "primary", primary_executables["status"],
         STATUS_PAIRS, [], True),
        ("normal", "primary", primary_executables["normal"],
         NORMAL_PAIRS, NORMAL_NOT_COMPARABLE, False),
        ("normal", "primary", primary_executables["normal"],
         NORMAL_PAIRS, NORMAL_NOT_COMPARABLE, True),
    ]
    if placement_executables is not None:
        runs.extend((
            ("status", "placement", placement_executables["status"],
             STATUS_PAIRS, [], False),
            ("status", "placement", placement_executables["status"],
             STATUS_PAIRS, [], True),
            ("normal", "placement", placement_executables["normal"],
             NORMAL_PAIRS, NORMAL_NOT_COMPARABLE, False),
            ("normal", "placement", placement_executables["normal"],
             NORMAL_PAIRS, NORMAL_NOT_COMPARABLE, True),
        ))
    try:
        apply_benchmark_execution_controls(
            args.affinity_mask, args.priority_class)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    log_records = []
    grouped_orders = {}
    for name, shape, executable, pairs, not_comparable, peer_first in runs:
        order = "peer-first" if peer_first else "candidate-first"
        command = bench_stat_command(
            bench_stat, executable, args.runs,
            args.equivalence_margin_pct, absl_dir, pairs, not_comparable,
            args.require_clean)
        environment = os.environ.copy()
        if peer_first:
            environment["HYLE_BENCH_RIVAL_FIRST"] = "1"
        else:
            environment.pop("HYLE_BENCH_RIVAL_FIRST", None)
        log_path = output_dir / f"{name}-{shape}-{order}.log"
        recorded_commands.append(command)
        text = run_and_capture(command, environment, log_path)
        raw_count, raw_records = validate_log(
            text, log_path, args.runs, len(pairs) + len(not_comparable))
        grouped_orders[(name, shape, order)] = validate_pair_records(
            raw_records, log_path, args.runs, pairs, not_comparable)
        log_records.append({
            "benchmark": name,
            "shape": shape,
            "order": order,
            "environment_overrides": {
                "HYLE_BENCH_RIVAL_FIRST": "1" if peer_first else None,
            },
            "path": str(log_path),
            "sha256": sha256(log_path),
            "raw_pair_records": raw_count,
            "comparable_pairs": len(pairs),
            "not_comparable_pairs": len(not_comparable),
        })

    input_hashes_after = collect_input_hashes(repo)
    if input_hashes_after != input_hashes_before:
        raise RuntimeError(
            "error benchmark/product inputs changed during build or sampling")

    bench_stat_module = load_bench_stat(bench_stat)
    dual_order_summary = summarize_dual_orders(
        bench_stat_module, grouped_orders, args.equivalence_margin_pct,
        args.placement)
    summary_path = output_dir / "dual-order-summary.json"
    summary_path.write_text(
        json.dumps({
            "record": "hyle_error_bench_dual_order_summary",
            "schema": 2,
            "ratio_definition": "RIVAL op/s / HYLE BASELINE op/s",
            "equivalence_margin_pct": args.equivalence_margin_pct,
            "placement_variant_available": args.placement,
            "placement_mechanism": (
                "MSVC LINK /FUNCTIONPADMIN:64" if args.placement else None),
            "platform_gate_complete": False,
            "pairs": dual_order_summary,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    manifest = {
        "record": "hyle_error_bench_evidence_manifest",
        "schema": 2,
        "observed_at_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor()
            or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        },
        "runs_per_order": args.runs,
        "equivalence_margin_pct": args.equivalence_margin_pct,
        "require_clean": args.require_clean,
        "fresh_configure": args.fresh_configure,
        "placement_variant": args.placement,
        "linkage_contract": {
            "hyle": "static",
            "abseil": "static",
            "same_final_pe": True,
            "claim_scope": "static shipping shape SOTA peer comparison",
        },
        "build_tree": str(build),
        "absl_dir": str(absl_dir),
        "source_git_at_start": source_git,
        "peer_git_at_start": peer_git,
        "input_files_sha256": input_hashes_before,
        "build_files_sha256": collect_build_bindings(build),
        "cmake_contract": cmake_contract,
        "cmake_version": command_text(["cmake", "--version"]).splitlines()[0],
        "python_version": platform.python_version(),
        "compiler": compiler_record(build),
        "execution_environment": execution_environment_record(
            args.affinity_mask, args.priority_class),
        "executables": {
            "primary": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in primary_executables.items()
            },
            "placement": None if placement_executables is None else {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in placement_executables.items()
            },
        },
        "commands": recorded_commands,
        "logs": log_records,
        "dual_order_summary": {
            "path": str(summary_path),
            "sha256": sha256(summary_path),
        },
        "code_shape": code_shape,
        "placement_layout": placement_layout,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"evidence manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
