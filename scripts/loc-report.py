#!/usr/bin/env python3
"""Report the four canonical HYLE module-closure LOC metrics.

The report deliberately counts the current contents of Git-tracked files.  It
therefore describes a dirty working tree honestly without letting untracked
build directories, caches, or scratch files change the result.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable, Sequence


NATIVE_LANGUAGES = ("C++", "C/C++ Header", "C", "CUDA")
SELF_WITH_TESTS_ROOTS = ("include/hyle", "src", "tests")
SELF_WITHOUT_TESTS_ROOTS = ("include/hyle", "src")

# Some closure units do not match the ordinary
# include/hyle/<module> + include/hyle/c/<module>*.h + src/<module> layout.
# Keep these semantic boundaries explicit: directory proximity is not module
# ownership.  The core list is the boundary recorded by its closure audit.
MODULE_BOUNDARIES = {
    "core": (
        "include/hyle/core",
        "include/hyle/c/core.h",
        "include/hyle/macros.h",
        "include/hyle/visibility.h",
        "src/core",
    ),
    "util-core": (
        "include/hyle/util/random.h",
        "include/hyle/util/ref_counted.h",
        "include/hyle/util/scope_guard.h",
        "include/hyle/util/spin_lock.h",
        "include/hyle/util/type_name.h",
    ),
    "util-runtime": (
        "include/hyle/util/demangle.h",
        "include/hyle/util/registry.h",
        "include/hyle/c/util.h",
        "src/util",
    ),
    "util": (
        "include/hyle/util",
        "include/hyle/c/util.h",
        "src/util",
    ),
}


class ReportError(RuntimeError):
    pass


def run(
    args: Sequence[str], root: Path, *, binary: bool = False
) -> str | bytes:
    try:
        completed = subprocess.run(
            list(args),
            cwd=root,
            check=False,
            capture_output=True,
            text=not binary,
            encoding=None if binary else "utf-8",
        )
    except OSError as exc:
        raise ReportError(f"cannot run {args[0]!r}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        raise ReportError(
            f"command failed ({completed.returncode}): {' '.join(args)}\n"
            f"{stderr.strip()}"
        )
    return completed.stdout


def repository_root(start: Path) -> Path:
    output = run(("git", "rev-parse", "--show-toplevel"), start)
    assert isinstance(output, str)
    return Path(output.strip()).resolve()


def posix_path(value: str) -> str:
    value = value.replace("\\", "/").rstrip("/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ReportError(f"scope path must be repository-relative: {value!r}")
    return path.as_posix()


def tracked_worktree_files(root: Path) -> tuple[list[str], list[str]]:
    output = run(("git", "ls-files", "-z"), root, binary=True)
    assert isinstance(output, bytes)
    try:
        names = output.decode("utf-8").split("\0")
    except UnicodeDecodeError as exc:
        raise ReportError("Git returned a tracked path that is not UTF-8") from exc

    files: list[str] = []
    missing: list[str] = []
    for raw_name in names:
        if not raw_name:
            continue
        name = posix_path(raw_name)
        if "\n" in name or "\r" in name:
            raise ReportError("cloc list files cannot represent newline-bearing paths")
        local = root.joinpath(*PurePosixPath(name).parts)
        if local.is_file():
            files.append(name)
        elif not local.is_dir():
            # A tracked file deleted in the working tree is not part of the
            # working snapshot.  Record it so the report remains auditable.
            missing.append(name)
    return sorted(set(files)), sorted(set(missing))


def under_roots(path: str, roots: Iterable[str]) -> bool:
    return any(path == root or path.startswith(root + "/") for root in roots)


def files_under(files: Iterable[str], roots: Sequence[str]) -> list[str]:
    normalized = tuple(posix_path(root) for root in roots)
    return [path for path in files if under_roots(path, normalized)]


def expand_specs(files: Sequence[str], specs: Sequence[str]) -> list[str]:
    selected: set[str] = set()
    for raw_spec in specs:
        spec = posix_path(raw_spec)
        if any(character in spec for character in "*?["):
            matches = [path for path in files if fnmatch.fnmatchcase(path, spec)]
        else:
            matches = [
                path
                for path in files
                if path == spec or path.startswith(spec + "/")
            ]
        if not matches:
            raise ReportError(f"module scope selects no tracked file: {raw_spec}")
        selected.update(matches)
    return sorted(selected)


def module_scope(module: str, files: Sequence[str]) -> tuple[list[str], list[str]]:
    normalized = module.strip().lower().replace("_", "-")
    if not normalized or "/" in normalized or "\\" in normalized:
        raise ReportError(f"invalid module name: {module!r}")

    if normalized in MODULE_BOUNDARIES:
        specs = list(MODULE_BOUNDARIES[normalized])
    else:
        directory_name = normalized.replace("-", "_")
        candidates = (
            f"include/hyle/{directory_name}",
            f"src/{directory_name}",
        )
        specs = [
            candidate
            for candidate in candidates
            if any(
                path == candidate or path.startswith(candidate + "/")
                for path in files
            )
        ]
        c_pattern = f"include/hyle/c/{directory_name}*.h"
        c_headers = sorted(
            path for path in files if fnmatch.fnmatchcase(path, c_pattern)
        )
        specs.extend(c_headers)
        if not specs:
            raise ReportError(
                f"no conventional implementation boundary found for module {module!r}; "
                "pass one or more --module-path values"
            )

    return specs, expand_specs(files, specs)


def custom_module_scope(
    raw_specs: Sequence[str], files: Sequence[str]
) -> tuple[list[str], list[str]]:
    specs = [posix_path(spec) for spec in raw_specs]
    return specs, expand_specs(files, specs)


def find_cloc() -> str:
    executable = shutil.which("cloc")
    if executable is None:
        raise ReportError(
            "cloc is required for the canonical report; install cloc and put it on PATH"
        )
    return executable


def cloc_version(cloc: str, root: Path) -> str:
    output = run((cloc, "--version"), root)
    assert isinstance(output, str)
    return output.strip().splitlines()[-1]


def cloc_measure(
    cloc: str,
    root: Path,
    files: Sequence[str],
    languages: Sequence[str] | None,
) -> tuple[int, int]:
    if not files:
        return 0, 0

    list_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", delete=False
        ) as list_file:
            list_name = list_file.name
            for path in files:
                list_file.write(path + "\n")

        args = [cloc, "--json", "--quiet", f"--list-file={list_name}"]
        if languages:
            args.append("--include-lang=" + ",".join(languages))
        output = run(args, root)
        assert isinstance(output, str)
        try:
            payload = json.loads(output)
            summary = payload.get("SUM", {})
            return int(summary.get("code", 0)), int(summary.get("nFiles", 0))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ReportError("cloc returned malformed JSON") from exc
    finally:
        if list_name:
            try:
                os.unlink(list_name)
            except FileNotFoundError:
                pass


def repository_identity(root: Path) -> tuple[str, bool]:
    commit_output = run(("git", "rev-parse", "HEAD"), root)
    status_output = run(
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        root,
        binary=True,
    )
    assert isinstance(commit_output, str)
    assert isinstance(status_output, bytes)
    return commit_output.strip(), bool(status_output)


def build_report(args: argparse.Namespace) -> dict[str, object]:
    root = repository_root(Path.cwd())
    tracked, missing = tracked_worktree_files(root)
    if not tracked:
        raise ReportError("the repository has no tracked working-tree files")

    if args.module_path:
        module_specs, module_files = custom_module_scope(args.module_path, tracked)
        boundary_kind = "explicit"
    else:
        module_specs, module_files = module_scope(args.module, tracked)
        boundary_kind = "canonical" if args.module in MODULE_BOUNDARIES else "conventional"

    self_with_tests = files_under(tracked, SELF_WITH_TESTS_ROOTS)
    self_without_tests = files_under(tracked, SELF_WITHOUT_TESTS_ROOTS)

    cloc = find_cloc()
    repository_code, repository_files = cloc_measure(cloc, root, tracked, None)
    self_with_tests_code, self_with_tests_files = cloc_measure(
        cloc, root, self_with_tests, NATIVE_LANGUAGES
    )
    self_without_tests_code, self_without_tests_files = cloc_measure(
        cloc, root, self_without_tests, NATIVE_LANGUAGES
    )
    module_code, module_code_files = cloc_measure(
        cloc, root, module_files, NATIVE_LANGUAGES
    )
    metrics = [
        {
            "id": "repository",
            "label": "full repository code",
            "code": repository_code,
            "files": repository_files,
            "selected_files": len(tracked),
            "paths": ["Git-tracked regular files (current working tree)"],
            "languages": ["all languages recognized by cloc"],
        },
        {
            "id": "self_with_tests",
            "label": "self-written including tests",
            "code": self_with_tests_code,
            "files": self_with_tests_files,
            "selected_files": len(self_with_tests),
            "paths": list(SELF_WITH_TESTS_ROOTS),
            "languages": list(NATIVE_LANGUAGES),
        },
        {
            "id": "self_without_tests",
            "label": "self-written excluding tests",
            "code": self_without_tests_code,
            "files": self_without_tests_files,
            "selected_files": len(self_without_tests),
            "paths": list(SELF_WITHOUT_TESTS_ROOTS),
            "languages": list(NATIVE_LANGUAGES),
        },
        {
            "id": "module",
            "label": f"current module implementation ({args.module})",
            "code": module_code,
            "files": module_code_files,
            "selected_files": len(module_files),
            "paths": module_specs,
            "languages": list(NATIVE_LANGUAGES),
        },
    ]

    commit, dirty = repository_identity(root)
    report: dict[str, object] = {
        "schema": 2,
        "repository": {
            "root": str(root),
            "commit": commit,
            "dirty": dirty,
            "tracked_regular_files": len(tracked),
            "tracked_missing_files": missing,
        },
        "policy": {
            "snapshot": "current contents of Git-tracked regular files",
            "untracked": "not counted; still makes dirty=true",
            "self_written_languages": list(NATIVE_LANGUAGES),
            "self_written_excludes": [
                "bench",
                "docs",
                "third_party",
                "samples",
                "cmake/build/package scripts",
                "all build/cache/scratch directories",
            ],
            "module_excludes": ["tests", "bench", "docs", "third_party", "CMake"],
        },
        "tool": {"cloc": cloc_version(cloc, root)},
        "module": {
            "name": args.module,
            "boundary": boundary_kind,
            "paths": module_specs,
        },
        "metrics": metrics,
    }
    return report


def self_check(report: dict[str, object]) -> None:
    expected_core = (
        "include/hyle/core",
        "include/hyle/c/core.h",
        "include/hyle/macros.h",
        "include/hyle/visibility.h",
        "src/core",
    )
    if MODULE_BOUNDARIES["core"] != expected_core:
        raise ReportError("the canonical core boundary has drifted")

    metrics = report["metrics"]
    assert isinstance(metrics, list)
    values = {metric["id"]: metric["code"] for metric in metrics}
    if not (
        values["repository"] >= values["self_with_tests"]
        >= values["self_without_tests"] >= values["module"] >= 0
    ):
        raise ReportError("LOC subset ordering is inconsistent")

    module = report["module"]
    assert isinstance(module, dict)
    module_paths = module["paths"]
    assert isinstance(module_paths, list)
    excluded_roots = ("tests", "bench", "docs", "third_party")
    if any(under_roots(posix_path(path), excluded_roots) for path in module_paths):
        raise ReportError("module implementation boundary includes an excluded tree")
    if any(not under_roots(posix_path(path), SELF_WITHOUT_TESTS_ROOTS) for path in module_paths):
        raise ReportError("module implementation boundary escapes include/hyle + src")


def print_human(report: dict[str, object], passed_self_check: bool) -> None:
    repository = report["repository"]
    tool = report["tool"]
    module = report["module"]
    metrics = report["metrics"]
    assert isinstance(repository, dict)
    assert isinstance(tool, dict)
    assert isinstance(module, dict)
    assert isinstance(metrics, list)

    print("HYLE canonical LOC report")
    print(f"root:   {repository['root']}")
    print(f"commit: {repository['commit']}")
    print(f"dirty:  {'yes' if repository['dirty'] else 'no'}")
    print(f"cloc:   {tool['cloc']}")
    if repository["tracked_missing_files"]:
        print(
            "tracked files absent from working tree: "
            f"{len(repository['tracked_missing_files'])} (not counted)"
        )
    print()
    print(f"{'metric':<42} {'code':>10} {'counted files':>15}")
    print(f"{'-' * 42} {'-' * 10} {'-' * 15}")
    for metric in metrics:
        print(f"{metric['label']:<42} {metric['code']:>10,} {metric['files']:>15,}")

    print()
    print("path/language policy:")
    print("  full repository: current Git-tracked files; every cloc language")
    print(
        "  self-written + tests: include/hyle, src, tests; "
        + ", ".join(NATIVE_LANGUAGES)
    )
    print(
        "  self-written - tests: include/hyle, src; "
        + ", ".join(NATIVE_LANGUAGES)
    )
    print(
        "  self/module exclude bench, docs, third_party, samples, CMake and "
        "build/cache/scratch trees"
    )
    print(f"  module boundary ({module['boundary']}):")
    for path in module["paths"]:
        print(f"    - {path}")
    if passed_self_check:
        print("self-check: PASS")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report the four canonical LOC metrics for a HYLE closure unit."
    )
    parser.add_argument("--module", required=True, help="closure unit name, for example core")
    parser.add_argument(
        "--module-path",
        action="append",
        default=[],
        metavar="REPO_PATH",
        help=(
            "replace the canonical/conventional module boundary with an explicit "
            "tracked file or directory; repeat as needed"
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="verify canonical boundaries and LOC subset invariants",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        report = build_report(args)
        if args.self_check:
            self_check(report)
            report["self_check"] = "pass"
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print_human(report, args.self_check)
        return 0
    except ReportError as exc:
        print(f"loc-report: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
