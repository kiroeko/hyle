#!/usr/bin/env python3
"""Inventory CLOSED-factor public surfaces and correctness-test ownership.

This is a discovery tool.  It deliberately does not infer semantic coverage or
turn test counts into a correctness verdict.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
REGISTRY = "docs/implementation/module-status.json"
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".inc", ".inl"}
TEST_SUFFIXES = {".c", ".cc", ".cpp", ".cxx"}
NATIVE_SUFFIXES = HEADER_SUFFIXES | TEST_SUFFIXES | {".cu", ".cuh", ".s", ".asm"}
TEST_DECLARATION = re.compile(
    rb"\b(?:TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(")
CROSS_CUTTING_HEADERS = {"include/hyle/obs/trace.h"}
CORE_ROOT_HEADERS = {"include/hyle/version.h", "include/hyle/visibility.h"}


class InventoryError(RuntimeError):
    pass


def _load_selector():
    path = SCRIPT_DIR / "validation-select.py"
    spec = importlib.util.spec_from_file_location("hyle_validation_select", path)
    if spec is None or spec.loader is None:
        raise InventoryError(f"cannot load central graph adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _decode(snapshot, path: str) -> str:
    data = snapshot.read(path)
    if data is None:
        raise InventoryError(f"snapshot is missing {path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise InventoryError(f"{path} is not UTF-8") from error


def _dependency_closure(graph: dict, roots: list[str]) -> set[str]:
    closure: set[str] = set()
    pending = list(roots)
    while pending:
        factor = pending.pop()
        if factor in closure:
            continue
        if factor not in graph["factors"]:
            raise InventoryError(f"product names unknown factor {factor!r}")
        closure.add(factor)
        pending.extend(graph["factors"][factor]["deps"])
    return closure


def closed_product_names(inventory: dict) -> list[str]:
    """Return public product roots whose complete factor closure is CLOSED."""
    return [
        row["product"] for row in inventory["products"]
        if row["all_factors_closed"]
    ]


def build_inventory(root: Path, revision: str) -> dict:
    selector = _load_selector()
    snapshot = selector.Snapshot(root, revision)
    graph = selector.project_graph(snapshot)
    registry = json.loads(_decode(snapshot, REGISTRY))
    records = {row["module"]: row for row in registry["modules"]}
    closed = [
        row["module"] for row in registry["modules"]
        if row["state"] == "CLOSED"
    ]
    closed_set = set(closed)
    issues: list[dict[str, str]] = []

    all_headers = sorted(
        path for path in snapshot.files
        if path.startswith("include/hyle/") and Path(path).suffix.lower() in HEADER_SUFFIXES
    )
    header_owners: dict[str, list[str]] = {name: [] for name in closed}
    cross_cutting_headers: list[str] = []
    for path in all_headers:
        if path in CROSS_CUTTING_HEADERS:
            cross_cutting_headers.append(path)
            continue
        owner = "core" if path in CORE_ROOT_HEADERS else selector.owner(path, graph)
        if owner in closed_set:
            header_owners[owner].append(path)
        elif owner is None:
            issues.append({"kind": "unowned_public_header", "path": path})

    registration_surfaces = {
        path: snapshot.read(path) or b""
        for path in snapshot.files
        if path.startswith(("scripts/", "cmake/", ".github/"))
    }

    factors: list[dict] = []
    total_test_lines = 0
    total_test_declarations = 0
    total_test_sources = 0
    for factor in closed:
        graph_row = graph["factors"].get(factor)
        if graph_row is None:
            issues.append({"kind": "closed_factor_missing_from_graph", "factor": factor})
            continue
        if graph_row["queued"]:
            issues.append({"kind": "closed_factor_marked_queued", "factor": factor})

        record = records[factor]
        record_path = record.get("record", "")
        if not record_path or snapshot.read(record_path) is None:
            issues.append({"kind": "missing_terminal_record", "factor": factor,
                           "path": record_path})
        performance_path = record.get("performance", {}).get("evidence", "")
        if performance_path and snapshot.read(performance_path) is None:
            issues.append({"kind": "missing_performance_evidence", "factor": factor,
                           "path": performance_path})

        test_fragment = f"tests/{factor}/CMakeLists.txt"
        fragment_data = snapshot.read(test_fragment)
        if fragment_data is None:
            issues.append({"kind": "missing_test_fragment", "factor": factor,
                           "path": test_fragment})
            fragment_text = ""
        else:
            fragment_text = fragment_data.decode("utf-8", errors="replace")

        test_sources = sorted(
            path for path in snapshot.files
            if path.startswith(f"tests/{factor}/")
            and Path(path).suffix.lower() in TEST_SUFFIXES
        )
        tests = []
        for path in test_sources:
            data = snapshot.read(path) or b""
            declarations = len(TEST_DECLARATION.findall(data))
            lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
            normal_registration = Path(path).name in fragment_text
            external_references = sorted(
                owner_path for owner_path, owner_data in registration_surfaces.items()
                if path.encode("utf-8") in owner_data
            )
            registered = normal_registration or bool(external_references)
            if not registered:
                issues.append({"kind": "unregistered_test_source", "factor": factor,
                               "path": path})
            tests.append({
                "path": path,
                "lines": lines,
                "test_declarations": declarations,
                "mentioned_by_test_fragment": normal_registration,
                "external_driver_references": external_references,
                "registered": registered,
            })
            total_test_lines += lines
            total_test_declarations += declarations
            total_test_sources += 1

        source_files = sorted(
            path for path in snapshot.files
            if path.startswith(f"src/{factor}/")
            and Path(path).suffix.lower() in NATIVE_SUFFIXES
        )
        factors.append({
            "factor": factor,
            "candidate": record.get("candidate"),
            "terminal_record": record_path,
            "performance_evidence": performance_path or None,
            "direct_dependencies": graph_row["deps"],
            "public_headers": header_owners[factor],
            "implementation_sources": source_files,
            "test_fragment": test_fragment,
            "test_sources": tests,
            "totals": {
                "public_headers": len(header_owners[factor]),
                "implementation_sources": len(source_files),
                "test_sources": len(test_sources),
                "test_lines": sum(row["lines"] for row in tests),
                "test_declarations": sum(row["test_declarations"] for row in tests),
            },
        })

    products = []
    closed_product_factor_union: set[str] = set()
    for name, roots in graph["products"].items():
        closure = _dependency_closure(graph, roots)
        all_factors_closed = closure <= closed_set
        if all_factors_closed:
            closed_product_factor_union.update(closure)
        products.append({
            "product": name,
            "factor_roots": roots,
            "factor_closure": sorted(closure),
            "all_factors_closed": all_factors_closed,
            "non_closed_factors": sorted(closure - closed_set),
        })
    for factor in sorted(closed_set - closed_product_factor_union):
        issues.append({
            "kind": "closed_factor_missing_from_closed_product",
            "factor": factor,
        })

    # Cross-factor public journeys are project-owned evidence.  Keep them out
    # of every factor's totals so their existence cannot make an individual
    # CLOSED factor appear more qualified than it is.
    project_test_fragment = "tests/correctness/CMakeLists.txt"
    project_fragment_data = snapshot.read(project_test_fragment)
    project_fragment_text = (
        project_fragment_data.decode("utf-8", errors="replace")
        if project_fragment_data is not None else ""
    )
    project_tests = []
    project_test_lines = 0
    project_test_declarations = 0
    for path in sorted(
        candidate for candidate in snapshot.files
        if candidate.startswith("tests/correctness/")
        and Path(candidate).suffix.lower() in TEST_SUFFIXES
    ):
        data = snapshot.read(path) or b""
        declarations = len(TEST_DECLARATION.findall(data))
        lines = data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)
        registered = Path(path).name in project_fragment_text
        if not registered:
            issues.append({"kind": "unregistered_project_test_source", "path": path})
        project_tests.append({
            "path": path,
            "lines": lines,
            "test_declarations": declarations,
            "test_fragment": project_test_fragment,
            "registered": registered,
        })
        project_test_lines += lines
        project_test_declarations += declarations

    return {
        "schema": "hyle.correctness-inventory/v1",
        "disclaimer": (
            "Discovery only: counts and reachability do not prove semantic correctness."
        ),
        "revision": {
            "requested": revision,
            "commit": snapshot.commit,
            "tree": snapshot.tree,
        },
        "summary": {
            "closed_factors": len(closed),
            "public_headers": sum(len(paths) for paths in header_owners.values()),
            "cross_cutting_public_headers": len(cross_cutting_headers),
            "test_sources": total_test_sources,
            "test_lines": total_test_lines,
            "test_declarations": total_test_declarations,
            "project_test_sources": len(project_tests),
            "project_test_lines": project_test_lines,
            "project_test_declarations": project_test_declarations,
            "issues": len(issues),
        },
        "cross_cutting_public_headers": cross_cutting_headers,
        "factors": factors,
        "products": products,
        "project_tests": project_tests,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SCRIPT_DIR.parent,
                        help="HYLE checkout (default: repository containing this script)")
    parser.add_argument("--revision", default="HEAD",
                        help="Git commit/ref, or WORKTREE (default: HEAD)")
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument("--output", type=Path,
                             help="write JSON here instead of stdout")
    output_mode.add_argument(
        "--print-closed-products", action="store_true",
        help=(
            "print a comma-separated list of public products whose complete "
            "factor closure is CLOSED instead of the JSON inventory"
        ),
    )
    parser.add_argument("--check", action="store_true",
                        help="return nonzero when ownership/registration issues exist")
    args = parser.parse_args()

    try:
        inventory = build_inventory(args.root.resolve(), args.revision)
    except (InventoryError, OSError, ValueError) as error:
        print(f"correctness inventory failed: {error}", file=sys.stderr)
        return 2

    if args.print_closed_products:
        payload = ",".join(closed_product_names(inventory)) + "\n"
    else:
        payload = json.dumps(inventory, indent=2, sort_keys=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    if args.check and inventory["issues"]:
        print(
            f"correctness inventory found {len(inventory['issues'])} issue(s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
