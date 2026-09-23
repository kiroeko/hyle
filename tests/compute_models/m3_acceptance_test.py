"""Validate the M3 model-chain receipt and the 64 x 4 coverage adjudication."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs/implementation/tasks/compute/llm-moe-coverage-matrix.json"
EVIDENCE = ROOT / "evidence/correctness/compute-llm-moe-m3/acceptance.json"
REFERENCE = ROOT / "evidence/correctness/compute-llm-moe-m3/b16-reference-receipt.json"
SOURCE_ID = "4068044d8571635d8a86f1e96a43dad104efc7089169065f735601e350008155"
FINAL_STATES = {
    "PASS",
    "BACKEND_REJECTED",
    "NUMERIC_INCOMPATIBLE",
    "UPSTREAM_MISSING",
    "VALIDATION_NOT_RUN",
}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    matrix = load(MATRIX)
    acceptance = load(EVIDENCE)
    reference = load(REFERENCE)

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
    assert dict(sorted(counts.items())) == acceptance["coverage_summary"]["states"]
    assert acceptance["coverage_summary"]["cells"] == 256
    assert acceptance["coverage_summary"]["unadjudicated"] == 0
    assert acceptance["provider_integration_gate"] == {
        "state": "PASS",
        "admitted_integrations_outstanding": 0,
        "evidence": "docs/implementation/tasks/compute/llm-moe-m2-terminal.md",
    }
    assert len(acceptance["groups"]) == 6
    assert all(group["state"] == "ADJUDICATED" for group in acceptance["groups"])

    assert acceptance["input"]["source_id"] == SOURCE_ID
    assert reference["source_id"] == SOURCE_ID
    assert reference["kind"] == "external_reference_execution_not_hyle_verdict"
    assert len(reference["cases"]) == 3
    assert all(case["result"] == "REFERENCE_OUTPUT_GENERATED" for case in reference["cases"])
    assert acceptance["local_model_journeys"]["hyle_passed"] == 0
    assert acceptance["local_model_journeys"]["reference_is_not_hyle_execution"] is True
    print(json.dumps({"requirements": 64, "cells": cells, "states": counts}, default=dict))


if __name__ == "__main__":
    main()
