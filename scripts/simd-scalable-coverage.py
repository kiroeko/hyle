#!/usr/bin/env python3
"""Audit fixed-width versus scalable SIMD definitions without confusing names with capability.

The historical per-header count is useful for finding drift, but it overstates the
SVE/RVV gap because shared vec.h generic operations and Native/Emulated selectors are
not backend-local definitions. This script keeps those dimensions separate and validates
the semantic classification in simd-scalable-op-manifest.json.

Run:
  python scripts/simd-scalable-coverage.py
  python scripts/simd-scalable-coverage.py --check
  python scripts/simd-scalable-coverage.py --json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
SIMD_INCLUDE = ROOT / "include" / "hyle" / "simd"
MANIFEST = ROOT / "scripts" / "simd-scalable-op-manifest.json"

FIXED = ("sse4", "avx2", "avx512", "neon")
SCALABLE = ("sve", "rvv")
ALL_BACKENDS = FIXED + ("scalar",) + SCALABLE
ALLOWED_CLASSES = {
    "shared_generic",
    "selector",
    "fixed_model",
    "extension_gated",
    "portable_missing",
}

DEF_RE = re.compile(
    r"HYLE_FORCE_INLINE\b[^(;{]*?\b([A-Za-z_][A-Za-z0-9_]*)\s*\("
)


def _strip_line_comment(line: str) -> str:
    return re.sub(r"//.*", "", line)


def extract_public_ops(path: pathlib.Path) -> set[str]:
    """Extract uppercase-leading inline definitions outside namespace detail."""
    result: set[str] = set()
    depth = 0
    detail_depth: int | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = _strip_line_comment(raw)
        if detail_depth is None and re.search(r"namespace\s+detail\b", line):
            detail_depth = depth
        if detail_depth is None:
            match = DEF_RE.search(line)
            if match:
                name = match.group(1)
                if name[0].isupper() and not re.fullmatch(r"[A-Z0-9_]+", name):
                    result.add(name)
        depth += line.count("{") - line.count("}")
        if detail_depth is not None and depth <= detail_depth:
            detail_depth = None
    return result


def load_manifest() -> dict:
    with MANIFEST.open(encoding="utf-8") as stream:
        return json.load(stream)


def build_report() -> tuple[dict, list[str]]:
    generic = extract_public_ops(SIMD_INCLUDE / "vec.h")
    local = {
        backend: extract_public_ops(SIMD_INCLUDE / "backend" / f"{backend}.h")
        for backend in ALL_BACKENDS
    }
    fixed_union = set().union(*(local[name] for name in FIXED))
    scalable_union = set().union(*(local[name] for name in SCALABLE))
    scalable_intersection = set.intersection(*(local[name] for name in SCALABLE))
    # A union-only audit hides asymmetric holes as soon as either scalable backend implements an
    # operation. Parity means each backend can instantiate the fixed semantic surface, so classify
    # every fixed op missing from at least one scalable backend.
    missing_by_backend = {name: fixed_union - local[name] for name in SCALABLE}
    missing_any = set().union(*missing_by_backend.values())

    manifest = load_manifest()
    errors: list[str] = []
    classes = manifest.get("classes", {})
    unknown_classes = set(classes) - ALLOWED_CLASSES
    missing_classes = ALLOWED_CLASSES - set(classes)
    if unknown_classes:
        errors.append(f"unknown manifest classes: {sorted(unknown_classes)}")
    if missing_classes:
        errors.append(f"missing manifest classes: {sorted(missing_classes)}")

    classified: dict[str, str] = {}
    duplicates: list[str] = []
    for class_name, entry in classes.items():
        for op in entry.get("ops", []):
            if op in classified:
                duplicates.append(op)
            classified[op] = class_name
    if duplicates:
        errors.append(f"ops classified more than once: {sorted(set(duplicates))}")

    manifest_ops = set(classified)
    if missing_any - manifest_ops:
        errors.append(
            "unclassified ops missing from a scalable backend: "
            f"{sorted(missing_any - manifest_ops)}"
        )
    if manifest_ops - missing_any:
        errors.append(
            "stale manifest ops present in both scalable backends: "
            f"{sorted(manifest_ops - missing_any)}"
        )

    generic_class = {
        op for op, class_name in classified.items() if class_name == "shared_generic"
    }
    bad_generic = generic_class - generic
    if bad_generic:
        errors.append(f"shared_generic ops not defined in vec.h: {sorted(bad_generic)}")

    selectors = {op for op, class_name in classified.items() if class_name == "selector"}
    malformed_selectors = {
        op for op in selectors if "Native" not in op and "Emulated" not in op
    }
    if malformed_selectors:
        errors.append(
            f"selector ops without Native/Emulated marker: {sorted(malformed_selectors)}"
        )

    product_dependencies = manifest.get("product_dependencies", {})
    for product, deps in product_dependencies.items():
        unknown = set(deps) - manifest_ops
        if unknown:
            errors.append(f"{product} dependencies outside manifest: {sorted(unknown)}")

    report = {
        "manifest_version": manifest.get("version"),
        "local_definition_counts": {name: len(local[name]) for name in ALL_BACKENDS},
        "shared_generic_count": len(generic),
        "syntactic_effective_counts": {
            name: len(local[name] | generic) for name in ALL_BACKENDS
        },
        "fixed_union_count": len(fixed_union),
        "scalable_union_count": len(scalable_union),
        "scalable_intersection_count": len(scalable_intersection),
        "missing_from_scalable_count": {
            name: len(missing_by_backend[name]) for name in SCALABLE
        },
        "missing_from_scalable": {
            name: sorted(missing_by_backend[name]) for name in SCALABLE
        },
        "missing_from_any_scalable_count": len(missing_any),
        "class_counts": {
            name: len(entry.get("ops", [])) for name, entry in classes.items()
        },
        "primary_implementation_backlog_count": len(
            classes.get("portable_missing", {}).get("ops", [])
        ),
        "product_dependencies": product_dependencies,
        "errors": errors,
    }
    return report, errors


def print_report(report: dict) -> None:
    print("SCALABLE SIMD COVERAGE (definitions != semantic capability)")
    print("\nBackend-local HYLE_FORCE_INLINE definitions:")
    for backend, count in report["local_definition_counts"].items():
        effective = report["syntactic_effective_counts"][backend]
        print(f"  {backend:8s} local={count:3d}  local|vec.h={effective:3d}")
    print(f"\nfixed union:       {report['fixed_union_count']}")
    print(f"scalable union:    {report['scalable_union_count']}")
    print(f"scalable common:   {report['scalable_intersection_count']}")
    for backend, count in report["missing_from_scalable_count"].items():
        print(f"fixed missing {backend:3s}: {count}")
    print(f"missing from any:  {report['missing_from_any_scalable_count']}")
    print("\nSemantic classification of names missing from at least one scalable backend:")
    for class_name, count in report["class_counts"].items():
        print(f"  {class_name:18s} {count:3d}")
    print(
        "\nPrimary register implementation backlog: "
        f"{report['primary_implementation_backlog_count']} portable_missing ops"
    )
    print("Product dependency seeds:")
    for product, deps in report["product_dependencies"].items():
        print(f"  {product:8s} {len(deps):2d}: {', '.join(deps)}")
    if report["errors"]:
        print("\nERRORS:")
        for error in report["errors"]:
            print(f"  - {error}")
    else:
        print("\nClassification is exhaustive and self-consistent.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="quiet success; fail on drift")
    parser.add_argument("--json", action="store_true", help="emit machine-readable report")
    args = parser.parse_args()

    report, errors = build_report()
    if args.json:
        json.dump(report, sys.stdout, indent=2, sort_keys=True)
        print()
    elif not args.check or errors:
        print_report(report)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
