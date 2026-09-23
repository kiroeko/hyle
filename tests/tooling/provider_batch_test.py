#!/usr/bin/env python3
"""Focused tests for mature-Provider batch governance."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).parents[2]
MANIFEST = ROOT / "docs/implementation/tasks/compute/llm-moe-m2-provider-batch.json"


def load_tool():
    script = ROOT / "scripts/provider-batch.py"
    spec = importlib.util.spec_from_file_location("hyle_provider_batch", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], text=True, capture_output=True,
        check=False)
    if completed.returncode != 0:
        raise AssertionError(completed.stderr or completed.stdout)
    return completed.stdout.strip()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Repository:
    def __init__(self, parent: Path, manifest: dict[str, object]) -> None:
        self.root = parent / "source"
        self.root.mkdir()
        git(self.root, "init", "-q")
        git(self.root, "config", "user.name", "HYLE Test")
        git(self.root, "config", "user.email", "hyle-test@example.invalid")
        (self.root / "seed.txt").write_text("baseline\n", encoding="utf-8")
        git(self.root, "add", "seed.txt")
        git(self.root, "commit", "-q", "-m", "baseline")
        self.baseline = git(self.root, "rev-parse", "HEAD")
        manifest["baseline"] = self.baseline
        self.manifest = self.root / "docs/implementation/tasks/compute/provider-batch.json"
        write_json(self.manifest, manifest)
        git(self.root, "add", self.manifest.relative_to(self.root).as_posix())
        git(self.root, "commit", "-q", "-m", "candidate")
        self.candidate = git(self.root, "rev-parse", "HEAD")
        self.tree = git(self.root, "rev-parse", "HEAD^{tree}")


class ProviderBatchTest(unittest.TestCase):
    def manifest(self) -> dict[str, object]:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_manifest_rejects_per_operation_profiles_and_unreasoned_skip(self) -> None:
        manifest = self.manifest()
        manifest["profiles"][0]["operation"] = "matmul"
        with self.assertRaisesRegex(TOOL.BatchError, "unknown fields"):
            TOOL.validate_manifest(manifest)

        manifest = self.manifest()
        manifest["gate_groups"] = [{
            "id": "b2-docs-na", "batch": "B2", "profile": "windows-source",
            "kind": "off", "requirements": ["F05.1"],
            "expected_selector_gates": [], "applicability": "not_applicable",
            "reason": "",
        }]
        with self.assertRaisesRegex(TOOL.BatchError, "requires a reason"):
            TOOL.validate_manifest(manifest)

    def test_plan_is_deterministic_low_parallel_and_candidate_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Repository(Path(temporary), self.manifest())
            first = TOOL.make_plan(repo.root, repo.manifest, repo.candidate)
            second = TOOL.make_plan(repo.root, repo.manifest, repo.candidate)
            self.assertEqual(first, second)
            self.assertEqual(first["candidate_commit"], repo.candidate)
            self.assertEqual(
                [row["route"] for row in first["profiles"]],
                ["all", "cpu", "cpu", "cuda", "cuda", "apple", "apple",
                 "vulkan", "vulkan"])
            for profile in first["profiles"]:
                self.assertEqual(profile["commands"]["build"][-2:], ["--parallel", "2"])
                for test in profile["commands"]["tests"]:
                    self.assertEqual(test[test.index("--parallel") + 1], "1")

            repo.manifest.write_text(
                repo.manifest.read_text(encoding="utf-8") + " ", encoding="utf-8")
            # Whitespace alone is not a semantic identity change.
            TOOL.make_plan(repo.root, repo.manifest, repo.candidate)
            changed = json.loads(repo.manifest.read_text(encoding="utf-8"))
            changed["description"] += " drift"
            write_json(repo.manifest, changed)
            with self.assertRaisesRegex(TOOL.BatchError, "does not match"):
                TOOL.make_plan(repo.root, repo.manifest, repo.candidate)

    def test_incremental_bundle_fetches_exact_candidate_and_rejects_dirty_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            repo = Repository(parent, self.manifest())
            bundle = parent / "candidate.bundle"
            receipt = TOOL.create_bundle(repo.root, repo.manifest, repo.candidate, bundle)
            self.assertEqual(receipt["candidate_commit"], repo.candidate)
            self.assertEqual(receipt["prerequisites"], [repo.baseline])
            temporary_ref = subprocess.run(
                ["git", "-C", str(repo.root), "show-ref", "--verify",
                 "refs/hyle-batch/candidate"], capture_output=True, check=False)
            self.assertNotEqual(temporary_ref.returncode, 0)

            receiver = parent / "receiver"
            receiver.mkdir()
            git(receiver, "init", "-q")
            git(receiver, "fetch", str(repo.root), repo.baseline)
            git(receiver, "fetch", str(bundle), "refs/hyle-batch/candidate")
            self.assertEqual(git(receiver, "rev-parse", "FETCH_HEAD"), repo.candidate)

            (repo.root / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            with self.assertRaisesRegex(TOOL.BatchError, "clean canonical worktree"):
                TOOL.create_bundle(repo.root, repo.manifest, repo.candidate,
                                   parent / "dirty.bundle")
            (repo.root / "dirty.txt").unlink()
            unrelated = git(repo.root, "commit-tree", repo.tree, "-m", "unrelated")
            with self.assertRaisesRegex(TOOL.BatchError, "not an ancestor"):
                TOOL.create_bundle(repo.root, repo.manifest, unrelated,
                                   parent / "unrelated.bundle")

    def test_finalize_requires_selector_pass_and_preserves_residual(self) -> None:
        manifest = self.manifest()
        manifest["gate_groups"] = [
            {
                "id": "b2-windows-cpu", "batch": "B2",
                "profile": "windows-cpu-shared", "kind": "correctness",
                "requirements": ["F05.1"], "expected_selector_gates": ["docs"],
                "applicability": "required", "reason": "B2 CPU acceptance",
            },
            {
                "id": "b2-source-resource-na", "batch": "B2",
                "profile": "windows-source", "kind": "resource",
                "requirements": ["F05.1"], "expected_selector_gates": [],
                "applicability": "not_applicable",
                "reason": "source packaging has no runtime resource lifecycle",
            },
        ]
        manifest["residuals"] = [{
            "id": "b2-vulkan-f05.1", "batch": "B2", "route": "vulkan",
            "requirements": ["F05.1"], "state": "UNPROVEN",
            "reason": "fixture residual", "owner": "M2-B2",
        }]
        with tempfile.TemporaryDirectory() as temporary:
            repo = Repository(Path(temporary), manifest)
            plan = TOOL.make_plan(repo.root, repo.manifest, repo.candidate)
            plan_path = repo.root / "plan.json"
            write_json(plan_path, plan)
            selector = repo.root / "selector.json"
            write_json(selector, {
                "schema_version": 1, "verdict": "PASS",
                "candidate_commit": repo.candidate, "candidate_tree": repo.tree,
                "selected_gates": [{"id": "docs", "status": "PASS"}],
                "skipped_gates": [],
            })
            ledger_path = repo.root / "ledger.json"
            ledger = {
                "schema_version": 1, "candidate_commit": repo.candidate,
                "candidate_tree": repo.tree,
                "runs": [{"gate_group": "b2-windows-cpu",
                          "selector_result": "selector.json",
                          "sha256": file_digest(selector)}],
            }
            write_json(ledger_path, ledger)
            terminal = TOOL.finalize(repo.manifest, plan_path, ledger_path)
            self.assertEqual(terminal["outcome"], "PASS_WITH_RECORDED_RESIDUALS")
            self.assertEqual(terminal["gate_groups"][1]["status"], "SKIPPED")
            self.assertEqual(terminal["residuals"][0]["state"], "UNPROVEN")
            self.assertEqual(terminal["gate_groups"][0]["selector_result"],
                             "selector.json")

            bad = copy.deepcopy(ledger)
            bad["runs"][0]["sha256"] = "0" * 64
            write_json(ledger_path, bad)
            with self.assertRaisesRegex(TOOL.BatchError, "digest mismatch"):
                TOOL.finalize(repo.manifest, plan_path, ledger_path)


if __name__ == "__main__":
    unittest.main()
