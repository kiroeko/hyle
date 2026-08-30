#!/usr/bin/env python3
"""Preserve certified SIMD peer reports as per-operation route evidence.

This tool does not time code and does not choose a provider. It consumes output
from op-parity-report.py, verifies that every report passed calibration and
certification, maps every measured title back to the frozen 387-operation
ledger, and writes a compact, hash-bound receipt. Route decisions remain a
separate semantic/product step so a valid INCONCLUSIVE row cannot silently turn
into a winner.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
COVERAGE_SCRIPT = SCRIPT_DIR / "simd-op-coverage.py"
SPEC = importlib.util.spec_from_file_location(
    "hyle_simd_op_coverage", COVERAGE_SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import SIMD coverage helper: {COVERAGE_SCRIPT}")
COVERAGE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = COVERAGE
SPEC.loader.exec_module(COVERAGE)

REPORT_HEADER = (
    "op\tratio_paired\tci95_lo\tci95_hi\tratio_med\tspreadH\tspreadG"
    "\tplace_u\tverdict"
)
VERDICTS = {"LOSE", "INCONCLUSIVE", "WIN", "TIE"}
RECEIPT_STATES = VERDICTS | {"INFO"}
REPORT_PRACTICAL_BAND_PCT = 3
ROUTE_MATERIAL_MARGIN_PCT = 5
PEER_MARKER_RE = re.compile(
    r"^# HYLE_OP_PEER peer=(HIGHWAY|XSIMD) revision=([0-9a-f]{40})\r?$",
    re.MULTILINE,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_shell_text(path: Path) -> str:
    """Decode output redirected by UTF-8 shells or Windows PowerShell 5.1."""
    raw = path.read_bytes()
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    return raw.decode(encoding, errors="replace")


def verify_peer_marker(path: Path, peer: str, revision: str) -> dict[str, object]:
    markers = PEER_MARKER_RE.findall(read_shell_text(path))
    expected = (peer, revision)
    if markers != [expected]:
        rendered = ", ".join(f"{found_peer}@{found_revision}" for found_peer, found_revision in markers)
        raise RuntimeError(
            f"{path}: expected exactly one peer marker {peer}@{revision}; "
            f"found {rendered or 'none'}")
    return {
        "name": path.name,
        "sha256": sha256(path),
        "peer_marker": {"peer": peer, "revision": revision, "verified": True},
    }


def all_operations() -> set[str]:
    result: set[str] = set()
    for header in COVERAGE.HEADERS:
        result |= COVERAGE.extract_public_ops(Path(COVERAGE.INC) / header)
    return result


def parse_report(path: Path, operations: set[str]) -> list[dict[str, object]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if "CALIBRATION PASS" not in text:
        raise RuntimeError(f"{path}: report did not pass calibration")
    if "CERTIFICATION PASS" not in text:
        raise RuntimeError(f"{path}: report did not pass certification")
    if REPORT_HEADER not in text:
        raise RuntimeError(f"{path}: report row schema is missing")

    rows: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line or line == REPORT_HEADER:
            continue
        if line.startswith("# ") and line.endswith("\tINFO"):
            fields = line[2:].split("\t")
            if len(fields) != 7:
                raise RuntimeError(
                    f"{path}: malformed replicated informational row: {line}")
            title = fields[0]
            mapped = sorted(
                COVERAGE._title_to_ops(title, operations) & operations)
            if not mapped:
                raise RuntimeError(
                    f"{path}: informational title has no frozen operation: {title}")
            rows.append({
                "title": title,
                "operations": mapped,
                "comparison_kind": "INFORMATIONAL",
                "ratio_a_over_b": float(fields[1]),
                "ci95_lo": float(fields[2]),
                "ci95_hi": float(fields[3]),
                "order_medians": fields[4],
                "sample_count": int(fields[5]),
                "verdict": "INFO",
            })
            continue
        if line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 9 or fields[-1] not in VERDICTS:
            continue
        title = fields[0]
        mapped = sorted(COVERAGE._title_to_ops(title, operations) & operations)
        if not mapped:
            raise RuntimeError(f"{path}: measured title has no frozen operation: {title}")
        rows.append({
            "title": title,
            "operations": mapped,
            "comparison_kind": "PEER",
            "ratio_hyle_over_peer": float(fields[1]),
            "ci95_lo": float(fields[2]),
            "ci95_hi": float(fields[3]),
            "ratio_median": float(fields[4]),
            "spread_hyle": float(fields[5]),
            "spread_peer": float(fields[6]),
            "placement_uncertainty": float(fields[7]),
            "verdict": fields[8],
        })
    if not rows:
        raise RuntimeError(f"{path}: no certified comparison rows found")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path,
                        default=SCRIPT_DIR / "simd-op-routing.json")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--raw-input", type=Path, action="append", default=[])
    parser.add_argument("--peer", choices=("HIGHWAY", "XSIMD"), required=True)
    parser.add_argument("--machine-profile", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--require-source-bench-coverage", action="store_true")
    args = parser.parse_args()

    try:
        ledger = json.loads(args.ledger.read_text(encoding="utf-8"))
        peer_revision = ledger["candidate_providers"][args.peer]
        if args.peer == "XSIMD" and not args.raw_input:
            raise RuntimeError(
                "XSIMD evidence requires raw inputs carrying the mandatory peer marker")
        raw_receipts = []
        for raw_input in args.raw_input:
            if args.peer == "XSIMD":
                raw_receipts.append(
                    verify_peer_marker(raw_input, args.peer, peer_revision))
            else:
                raw_receipts.append({
                    "name": raw_input.name,
                    "sha256": sha256(raw_input),
                    "peer_marker": {"required": False},
                })
        operations = all_operations()
        rows: list[dict[str, object]] = []
        report_receipts = []
        for report in args.report:
            report_rows = parse_report(report, operations)
            rows.extend(report_rows)
            report_receipts.append({
                "name": report.name,
                "sha256": sha256(report),
                "row_count": len(report_rows),
            })

        observed = {
            operation
            for row in rows
            for operation in row["operations"]
        }
        source_benched = COVERAGE.extract_benched(operations)
        missing = sorted(source_benched - observed)
        if args.require_source_bench_coverage and missing:
            raise RuntimeError(
                f"certified report union misses {len(missing)} source-benched "
                f"operations: {', '.join(missing)}")

        verdict_counts = {
            verdict: sum(row["verdict"] == verdict for row in rows)
            for verdict in sorted(RECEIPT_STATES)
        }
        operation_rows = {
            operation: [
                index for index, row in enumerate(rows)
                if operation in row["operations"]
            ]
            for operation in sorted(observed)
        }
        try:
            ledger_path = args.ledger.resolve().relative_to(Path(COVERAGE.ROOT).resolve())
            ledger_name = ledger_path.as_posix()
        except ValueError:
            ledger_name = args.ledger.name
        receipt = {
            "schema": "hyle.simd.route-qualification/v2",
            "verdict": "PASS",
            "decision_state": "EVIDENCE_ONLY",
            "peer": args.peer,
            "peer_revision": peer_revision,
            "ratio_orientation": "HYLE/peer",
            "machine_profile": args.machine_profile,
            "target": args.target,
            "compiler": args.compiler,
            "ledger": {
                # Bind evidence to the immutable frozen projection, not the whole mutable route
                # ledger.  SIMD-15 is intentionally incremental: signing a later operation must
                # not invalidate measurements whose operation/source inventory did not change.
                "path": ledger_name,
                "schema": ledger["schema"],
                "baseline_commit": ledger["baseline"]["commit"],
                "inventory_sha256": ledger["baseline"]["inventory_sha256"],
                "operation_count": ledger["baseline"]["operation_count"],
            },
            "instrument": {
                "engine_sha256": sha256(
                    Path(COVERAGE.ROOT) / "bench/simd/op_parity_engine.h"),
                "reporter_sha256": sha256(SCRIPT_DIR / "op-parity-report.py"),
                # The reporter's narrower band says whether the instrument resolved a
                # difference.  SIMD's frozen P1 margin separately says whether that
                # difference is material enough to change a product route.
                "report_practical_band_pct": REPORT_PRACTICAL_BAND_PCT,
                "route_material_margin_pct": ROUTE_MATERIAL_MARGIN_PCT,
            },
            "reports": report_receipts,
            "raw_inputs": raw_receipts,
            "row_count": len(rows),
            "verdict_counts": verdict_counts,
            "source_benched_operation_count": len(source_benched),
            "observed_operation_count": len(observed),
            "missing_source_benched_operations": missing,
            "operation_rows": operation_rows,
            "rows": rows,
        }
        args.evidence.parent.mkdir(parents=True, exist_ok=True)
        args.evidence.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
    except (OSError, KeyError, ValueError, json.JSONDecodeError,
            RuntimeError) as error:
        print(f"SIMD route qualification receipt FAILED: {error}", file=sys.stderr)
        return 1

    print(
        "SIMD route qualification receipt OK: "
        f"{len(observed)}/{len(source_benched)} source-benched operations, "
        f"{len(rows)} rows, peer={args.peer}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
