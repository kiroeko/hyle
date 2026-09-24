"""Structural checks for the batch D5 real-profile acceptance ledger."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/implementation/tasks/compute"
LEDGER = DOCS / "llm-moe-d5-acceptance-ledger.json"
DEMAND = DOCS / "llm-moe-provider-demand-matrix.json"
SOURCES = DOCS / "llm-moe-model-sources.json"

ROUTES = {"cpu", "cuda", "apple", "vulkan"}
ENTRIES = {"direct", "graph"}
STATES = {
    "RUN_PASS",
    "BACKEND_REJECTED",
    "UPSTREAM_MISSING",
    "VALIDATION_NOT_RUN",
}
REGIONS = {f"R{value:02d}" for value in range(1, 16)}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ledger = load(LEDGER)
    demand = load(DEMAND)
    sources = load(SOURCES)

    source_models = {item["id"] for item in sources["models"]}
    expected_profiles = {}
    for item in demand["profile_catalog"]:
        model = item["model"]
        for role in ("official", "primary_gguf", "fidelity_gguf"):
            profile = item.get(role)
            if profile:
                identity = f"{model}:{role}:{profile['variant']}"
                expected_profiles[identity] = (model, role, profile)

    profiles = ledger["profiles"]
    assert len(profiles) == len(expected_profiles) == 22
    assert len({item["id"] for item in profiles}) == len(profiles)
    assert {item["model"] for item in profiles} == source_models
    for item in profiles:
        model, role, expected = expected_profiles[item["id"]]
        assert item["model"] == model
        assert item["role"] == role
        for field in ("repository", "revision", "variant", "formats"):
            assert item[field] == expected[field]
        assert item["regions"]
        assert set(item["regions"]) <= REGIONS
        assert item["shape_anchor"]
        assert all(isinstance(value, int) and value > 0
                   for value in item["shape_anchor"].values())

    cells = ledger["cells"]
    assert len(cells) == len(profiles) * len(ROUTES) * len(ENTRIES) == 176
    assert len({item["id"] for item in cells}) == len(cells)
    profile_ids = {item["id"] for item in profiles}
    profile_regions = {item["id"]: set(item["regions"]) for item in profiles}
    observed = set()
    states: Counter[str] = Counter()
    for item in cells:
        assert item["profile"] in profile_ids
        assert item["route"] in ROUTES
        assert item["entry"] in ENTRIES
        assert item["status"] in STATES
        key = (item["profile"], item["route"], item["entry"])
        assert key not in observed
        observed.add(key)
        states[item["status"]] += 1
        partial_regions = set()
        for partial in item.get("partial_receipts", []):
            assert partial["verdict"] in {"RUN_PASS", "BACKEND_REJECTED",
                                           "UPSTREAM_MISSING"}
            regions = set(partial["regions"])
            assert regions and regions <= profile_regions[item["profile"]]
            assert not (partial_regions & regions)
            partial_regions |= regions
            local = ROOT / partial["receipt"].split("#", 1)[0]
            assert local.is_file(), partial["receipt"]
        if item["status"] == "VALIDATION_NOT_RUN":
            assert item["receipt"] is None
            assert not item["region_verdicts"]
        else:
            assert item["receipt"]
            local = ROOT / item["receipt"].split("#", 1)[0]
            assert local.is_file(), item["receipt"]
            assert item["region_verdicts"]

    counts = ledger["counts"]
    assert counts["models"] == 8
    assert counts["profiles"] == 22
    assert counts["routes"] == 4
    assert counts["entries"] == 2
    assert counts["cells"] == 176
    for state in STATES:
        assert counts[state] == states[state]
    print(json.dumps({"profiles": len(profiles), "cells": len(cells),
                      "states": dict(sorted(states.items()))}))


if __name__ == "__main__":
    main()
