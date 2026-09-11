#!/usr/bin/env python3
"""Run or collect GCC line/branch coverage for selected CLOSED factors.

Coverage is a discovery instrument: the report highlights executable zero-hit
paths and files absent from this build, but never emits a correctness verdict.
Run this script on Linux/WSL with GCC 14 and an exclusive fresh build directory.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = SCRIPT_DIR.parent
REGISTRY = "docs/implementation/module-status.json"
INVENTORY_SCRIPT = SCRIPT_DIR / "correctness-inventory.py"


class CoverageError(RuntimeError):
    pass


def _line_ranges(lines: list[int]) -> list[str]:
    if not lines:
        return []
    values = sorted(set(lines))
    result: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        result.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    result.append(str(start) if start == previous else f"{start}-{previous}")
    return result


def _closed_factors(root: Path) -> set[str]:
    registry = json.loads((root / REGISTRY).read_text(encoding="utf-8"))
    return {
        row["module"] for row in registry["modules"]
        if row["state"] == "CLOSED"
    }


def _factor_list(root: Path, raw: str) -> list[str]:
    factors = list(dict.fromkeys(
        item.strip() for item in re.split(r"[,;]", raw) if item.strip()
    ))
    if not factors:
        raise CoverageError("--factors must name at least one CLOSED factor")
    closed = _closed_factors(root)
    invalid = sorted(set(factors) - closed)
    if invalid:
        raise CoverageError(
            "coverage accepts only current CLOSED factors; rejected: "
            + ", ".join(invalid)
        )
    return factors


def _inventory(root: Path) -> dict:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "hyle_correctness_inventory_for_coverage", INVENTORY_SCRIPT)
    if spec is None or spec.loader is None:
        raise CoverageError(f"cannot load {INVENTORY_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.build_inventory(root, "WORKTREE")


def _source_binding(root: Path, factors: list[str]) -> dict:
    report = _inventory(root)
    selected = set(factors)
    expected: dict[str, str] = {}
    for row in report["factors"]:
        if row["factor"] not in selected:
            continue
        for path in row["public_headers"] + row["implementation_sources"]:
            expected[path] = row["factor"]
    return {
        "schema": "hyle.correctness-coverage-input/v1",
        "source_revision": report["revision"],
        "factors": factors,
        "product_files": {
            path: {
                "factor": factor,
                "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest(),
            }
            for path, factor in sorted(expected.items())
        },
    }


def _load_binding(build_dir: Path, factors: list[str]) -> tuple[dict[str, str], dict]:
    path = build_dir / "hyle-correctness-coverage-input.json"
    if not path.is_file():
        raise CoverageError(
            f"missing run-mode source binding: {path}; collect never attributes an "
            "unbound build to the current worktree"
        )
    binding = json.loads(path.read_text(encoding="utf-8"))
    if binding.get("schema") != "hyle.correctness-coverage-input/v1":
        raise CoverageError(f"unsupported source binding in {path}")
    if binding.get("factors") != factors:
        raise CoverageError(
            "collect factors differ from the run-mode source binding: "
            f"{binding.get('factors')!r} != {factors!r}"
        )
    product_files = binding.get("product_files")
    if not isinstance(product_files, dict) or not product_files:
        raise CoverageError(f"source binding has no product files: {path}")
    return {
        name: row["factor"] for name, row in product_files.items()
    }, binding["source_revision"]


def _relative_product_path(root: Path, value: str, cwd: str) -> str | None:
    path = Path(value)
    if not path.is_absolute():
        path = Path(cwd) / path
    try:
        relative = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return None
    if relative.startswith("include/hyle/") or relative.startswith("src/"):
        return relative
    return None


def aggregate_gcov(
    root: Path,
    payloads: list[dict],
    factors: list[str],
    expected: dict[str, str],
) -> dict:
    files: dict[str, dict] = {}
    for payload in payloads:
        cwd = payload.get("current_working_directory", str(root))
        for source in payload.get("files", []):
            path = _relative_product_path(root, source["file"], cwd)
            if path is None or path not in expected:
                continue
            merged = files.setdefault(path, {"lines": {}, "branches": {}, "functions": {}})
            for line in source.get("lines", []):
                number = int(line["line_number"])
                merged["lines"][number] = (
                    merged["lines"].get(number, 0) + int(line.get("count", 0))
                )
                for branch in line.get("branches", []):
                    key = (
                        number,
                        int(branch.get("source_block_id", -1)),
                        int(branch.get("destination_block_id", -1)),
                        bool(branch.get("fallthrough", False)),
                        bool(branch.get("throw", False)),
                    )
                    merged["branches"][key] = (
                        merged["branches"].get(key, 0) + int(branch.get("count", 0))
                    )
            for function in source.get("functions", []):
                key = (
                    function.get("demangled_name") or function.get("name", ""),
                    int(function.get("start_line", 0)),
                )
                merged["functions"][key] = (
                    merged["functions"].get(key, 0)
                    + int(function.get("execution_count", 0))
                )

    rows = []
    totals = {
        "executable_lines": 0,
        "covered_lines": 0,
        "branches": 0,
        "covered_branches": 0,
        "functions": 0,
        "covered_functions": 0,
    }
    for path in sorted(files):
        data = files[path]
        executable_lines = len(data["lines"])
        covered_lines = sum(count > 0 for count in data["lines"].values())
        branches = len(data["branches"])
        covered_branches = sum(count > 0 for count in data["branches"].values())
        functions = len(data["functions"])
        covered_functions = sum(count > 0 for count in data["functions"].values())
        zero_functions = sorted(
            name for (name, _), count in data["functions"].items()
            if count == 0 and name
        )
        rows.append({
            "path": path,
            "factor": expected[path],
            "executable_lines": executable_lines,
            "covered_lines": covered_lines,
            "line_coverage_pct": (
                round(100.0 * covered_lines / executable_lines, 2)
                if executable_lines else None
            ),
            "zero_hit_line_ranges": _line_ranges([
                number for number, count in data["lines"].items() if count == 0
            ]),
            "branches": branches,
            "covered_branches": covered_branches,
            "branch_coverage_pct": (
                round(100.0 * covered_branches / branches, 2) if branches else None
            ),
            "functions": functions,
            "covered_functions": covered_functions,
            "zero_hit_functions": zero_functions,
        })
        for key, value in (
            ("executable_lines", executable_lines),
            ("covered_lines", covered_lines),
            ("branches", branches),
            ("covered_branches", covered_branches),
            ("functions", functions),
            ("covered_functions", covered_functions),
        ):
            totals[key] += value

    observed = set(files)
    totals.update({
        "observed_product_files": len(observed),
        "expected_product_files": len(expected),
        "unobserved_product_files": len(set(expected) - observed),
        "line_coverage_pct": (
            round(100.0 * totals["covered_lines"] / totals["executable_lines"], 2)
            if totals["executable_lines"] else None
        ),
        "branch_coverage_pct": (
            round(100.0 * totals["covered_branches"] / totals["branches"], 2)
            if totals["branches"] else None
        ),
    })
    return {
        "schema": "hyle.correctness-coverage-discovery/v1",
        "verdict": "DISCOVERY_ONLY",
        "disclaimer": (
            "Coverage and test counts reveal unobserved paths; they do not prove "
            "semantic correctness or change module state."
        ),
        "factors": factors,
        "summary": totals,
        "unobserved_product_files": sorted(set(expected) - observed),
        "files": rows,
    }


def _load_payloads(raw_dir: Path) -> list[dict]:
    paths = sorted(raw_dir.glob("*.gcov.json.gz"))
    if not paths:
        raise CoverageError(f"gcov produced no JSON payloads in {raw_dir}")
    payloads = []
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payloads.append(json.load(stream))
    return payloads


def collect(
    root: Path,
    build_dir: Path,
    output: Path,
    factors: list[str],
    gcov: str,
    timeout: int,
) -> dict:
    gcno = sorted(
        path for path in build_dir.rglob("*.gcno")
        if "_deps" not in path.parts and "CompilerId" not in path.parts
    )
    if not gcno:
        raise CoverageError(f"no .gcno files found under {build_dir}")
    if shutil.which(gcov) is None:
        raise CoverageError(f"required tool is not available: {gcov}")
    raw_dir = output.parent / (output.stem + "-gcov-json")
    if raw_dir.exists() and any(raw_dir.iterdir()):
        raise CoverageError(f"refusing non-empty gcov output directory: {raw_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    gcov_log = output.parent / (output.stem + "-gcov.log")
    started = time.monotonic()
    with gcov_log.open("wb") as log:
        for offset in range(0, len(gcno), 40):
            remaining = timeout - int(time.monotonic() - started)
            if remaining <= 0:
                raise CoverageError(f"gcov collection exceeded {timeout}s")
            command = [
                gcov,
                "--json-format",
                "--branch-probabilities",
                "--branch-counts",
                "--hash-filenames",
                *[str(path) for path in gcno[offset:offset + 40]],
            ]
            try:
                result = subprocess.run(
                    command,
                    cwd=raw_dir,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=remaining,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise CoverageError(f"gcov collection exceeded {timeout}s") from error
            if result.returncode:
                raise CoverageError(
                    f"gcov failed with exit code {result.returncode}; see {gcov_log}"
                )
    expected, source_revision = _load_binding(build_dir, factors)
    report = aggregate_gcov(
        root, _load_payloads(raw_dir), factors, expected)
    report["inputs"] = {
        "root": str(root),
        "source_revision": source_revision,
        "build_dir": str(build_dir),
        "gcov": shutil.which(gcov) or gcov,
        "gcov_version": subprocess.check_output(
            [gcov, "--version"], text=True).splitlines()[0],
        "gcno_files": len(gcno),
        "raw_payloads": len(list(raw_dir.glob("*.gcov.json.gz"))),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _run_logged(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log: Path,
    timeout: int,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as stream:
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise CoverageError(
                f"{command[0]} exceeded {timeout}s; see {log}"
            ) from error
    if result.returncode:
        raise CoverageError(
            f"{command[0]} failed with exit code {result.returncode}; see {log}"
        )


def run_coverage(args, root: Path, factors: list[str]) -> dict:
    if os.name == "nt" or sys.platform != "linux":
        raise CoverageError("run mode requires Linux/WSL with GCC 14")
    for tool in (args.cmake, args.gcc, args.gxx, args.gcov):
        if shutil.which(tool) is None:
            raise CoverageError(f"required tool is not available: {tool}")

    build_dir = args.build_dir.resolve()
    if build_dir.exists() and any(build_dir.iterdir()):
        raise CoverageError(f"refusing non-empty build directory: {build_dir}")
    build_dir.mkdir(parents=True, exist_ok=True)
    binding = _source_binding(root, factors)
    (build_dir / "hyle-correctness-coverage-input.json").write_text(
        json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    output = args.output.resolve()
    logs = output.parent / (output.stem + "-logs")
    env = dict(os.environ, CC=args.gcc, CXX=args.gxx)
    default_cache = root.parent / ".deps"
    if "HYLE_DEP_SOURCE_CACHE" not in env and default_cache.exists():
        env["HYLE_DEP_SOURCE_CACHE"] = str(default_cache)

    coverage_flags = "--coverage -fprofile-abs-path -O0 -g"
    configure = [
        args.cmake, "-S", str(root), "-B", str(build_dir), "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Debug",
        "-DHYLE_DEV_FACTORS=" + ",".join(factors),
        "-DHYLE_BUILD_TESTS=ON",
        "-DHYLE_TEST_INSTRUMENTATION=" + (
            "ON" if args.instrumentation == "fault" else "OFF"
        ),
        "-DHYLE_BUILD_BENCH=OFF",
        "-DHYLE_INSTALL=OFF",
        "-DHYLE_WERROR=ON",
        "-DCMAKE_C_FLAGS=" + coverage_flags,
        "-DCMAKE_CXX_FLAGS=" + coverage_flags,
        "-DCMAKE_EXE_LINKER_FLAGS=--coverage",
        "-DCMAKE_SHARED_LINKER_FLAGS=--coverage",
    ]
    _run_logged(configure, root, env, logs / "configure.log", args.phase_timeout)
    _run_logged(
        [args.cmake, "--build", str(build_dir), "--target", "hyle_tests",
         "--parallel", str(args.parallel)],
        root,
        env,
        logs / "build.log",
        args.phase_timeout,
    )
    label = "^(" + "|".join(re.escape(name) for name in factors) + ")$"
    _run_logged(
        ["ctest", "--test-dir", str(build_dir), "--output-on-failure",
         "--no-tests=error", "--timeout", str(args.test_timeout), "-L", label,
         "--parallel", str(args.parallel)],
        root,
        env,
        logs / "ctest.log",
        args.phase_timeout,
    )
    report = collect(
        root, build_dir, output, factors, args.gcov, args.phase_timeout)
    report["run"] = {
        "instrumentation": args.instrumentation,
        "parallel": args.parallel,
        "per_test_timeout_seconds": args.test_timeout,
        "phase_timeout_seconds": args.phase_timeout,
        "logs": str(logs),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="HYLE checkout (default: repository containing this script)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="re-collect a GCC build previously prepared by run mode")
    collect_parser.add_argument("--build-dir", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--factors", required=True)
    collect_parser.add_argument("--gcov", default="gcov-14")
    collect_parser.add_argument("--phase-timeout", type=int, default=1800)

    run_parser = subparsers.add_parser(
        "run", help="configure a fresh build, run factor tests, then collect")
    run_parser.add_argument("--build-dir", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    run_parser.add_argument("--factors", required=True)
    run_parser.add_argument("--instrumentation", choices=("shipping", "fault"),
                            default="shipping")
    run_parser.add_argument("--parallel", type=int, default=2)
    run_parser.add_argument("--test-timeout", type=int, default=120)
    run_parser.add_argument("--phase-timeout", type=int, default=1800)
    run_parser.add_argument("--cmake", default="cmake")
    run_parser.add_argument("--gcc", default="gcc-14")
    run_parser.add_argument("--gxx", default="g++-14")
    run_parser.add_argument("--gcov", default="gcov-14")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        factors = _factor_list(root, args.factors)
        if args.command == "run":
            report = run_coverage(args, root, factors)
        else:
            report = collect(
                root,
                args.build_dir.resolve(),
                args.output.resolve(),
                factors,
                args.gcov,
                args.phase_timeout,
            )
    except (CoverageError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"correctness coverage failed: {error}", file=sys.stderr)
        return 2
    summary = report["summary"]
    print(
        "correctness coverage discovery: "
        f"{summary['covered_lines']}/{summary['executable_lines']} lines, "
        f"{summary['covered_branches']}/{summary['branches']} branches, "
        f"{summary['unobserved_product_files']} unobserved product files"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
