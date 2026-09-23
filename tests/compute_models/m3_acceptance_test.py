"""Validate the M3 model-chain receipt and the 64 x 4 coverage adjudication."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs/implementation/tasks/compute/llm-moe-coverage-matrix.json"
EVIDENCE = ROOT / "evidence/correctness/compute-llm-moe-m3/acceptance.json"
REFERENCE = ROOT / "evidence/correctness/compute-llm-moe-m3/b16-reference-receipt.json"
EXECUTION_MAP = ROOT / "evidence/correctness/compute-llm-moe-m3/execution-map.json"
SOURCE_ID = "4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155"
FINAL_STATES = {
    "PASS",
    "BACKEND_REJECTED",
    "NUMERIC_INCOMPATIBLE",
    "UPSTREAM_MISSING",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    matrix = load(MATRIX)
    acceptance = load(EVIDENCE)
    reference = load(REFERENCE)
    execution = load(EXECUTION_MAP)

    requirements = matrix["requirements"]
    assert matrix["requirement_count"] == 64 == len(requirements)
    assert len({row["id"] for row in requirements}) == 64

    counts: Counter[str] = Counter()
    cells = 0
    for row in requirements:
        assert set(row["route_coverage"]) == {
            "cpu", "cuda", "apple_mlx_metal", "vulkan"
        }
        for coverage in row["route_coverage"].values():
            cells += 1
            state = coverage["state"]
            assert state in FINAL_STATES
            assert coverage.get("reason")
            assert coverage.get("evidence")
            for item in coverage["evidence"]:
                local = ROOT / item.split("#", 1)[0]
                assert local.is_file(), item
            counts[state] += 1

    assert cells == 256
    assert counts["PASS"] == 153
    assert counts.get("VALIDATION_NOT_RUN", 0) == 0
    assert dict(sorted(counts.items())) == acceptance["coverage_summary"]["states"]
    assert acceptance["coverage_summary"]["cells"] == 256
    assert acceptance["coverage_summary"]["unadjudicated"] == 0
    assert acceptance["verdict"] == "M3_CANDIDATE"
    assert acceptance["provider_integration_gate"] == {
        "state": "PASS",
        "admitted_integrations_outstanding": 0,
        "evidence": "docs/implementation/tasks/compute/llm-moe-m2-terminal.md",
    }
    assert len(acceptance["groups"]) == 6
    assert all(group["state"] == "CANDIDATE" for group in acceptance["groups"])

    assert execution["resumed_cell_count"] == 134
    assert len(execution["resumed_cells"]) == 134
    assert len(set(execution["resumed_cells"])) == 134
    assert execution["validation_not_run"] == 0
    assert execution["final_states"] == dict(sorted(counts.items()))
    completed: dict[str, set[str]] = {}
    for receipt, metadata in execution["receipts"].items():
        path = ROOT / metadata["path"]
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == metadata["sha256"]
        report = json.loads(payload)
        names: set[str] = set()
        failures = 0
        for suite in report.get("testsuites", []):
            for case in suite.get("testsuite", []):
                names.add(f"{case.get('classname', '')}.{case['name']}")
                failures += len(case.get("failures", []))
        assert len(names) == metadata["tests"]
        assert failures == metadata["failures"] == 0
        completed[receipt] = names
    for cell, adjudication in execution["cells"].items():
        requirement, route = cell.rsplit(".", 1)
        row = next(row for row in requirements if row["id"] == requirement)
        assert row["route_coverage"][route]["state"] == adjudication["state"]
        refs = []
        for key in ("direct", "graph", "boundary", "positive_domain"):
            refs.extend(adjudication.get(key, []))
        assert refs
        for item in refs:
            assert item["test"] in completed[item["receipt"]], item

    assert acceptance["input"]["source_id"] == SOURCE_ID
    assert reference["source_id"] == SOURCE_ID
    assert reference["kind"] == "external_reference_execution_not_hyle_verdict"
    assert len(reference["cases"]) == 3
    assert all(case["result"] == "REFERENCE_OUTPUT_GENERATED" for case in reference["cases"])
    assert acceptance["local_model_journeys"]["hyle_passed"] == 0
    assert acceptance["local_model_journeys"]["reference_is_not_hyle_execution"] is True
    print(json.dumps({"requirements": 64, "cells": cells,
                      "resumed_cells": 134, "states": counts}, default=dict))


if __name__ == "__main__":
    main()
