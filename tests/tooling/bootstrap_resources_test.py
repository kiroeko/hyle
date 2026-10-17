"""Small host-tool preparation checks; no compiler, download, or SDK required."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import hyle_resources as resources


class BootstrapResourcesTest(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".scratch/validation/bootstrap-resources"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.directory = Path(self.temp.name)
        self.repo = self.directory / "repo"
        (self.repo / "cmake").mkdir(parents=True)
        self.lock = {"dependencies": [{
            "name": "glslang", "revision": "a" * 40,
            "archive": {"sha256": "b" * 64, "url": "https://invalid.test/source"},
        }]}
        self.write_lock()
        self.root_patch = patch.object(resources, "ROOT", self.repo)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        self.addCleanup(self.temp.cleanup)
        self.environment = SimpleNamespace(
            env={"PATH": ""}, system="Windows" if os.name == "nt" else "Linux",
            arch="x86_64", cmake="cmake", ctest="ctest", cc="cc", cxx="c++",
            generator="Ninja", definitions={},
        )
        self.cache = self.directory / "dependencies/v1"
        self.commands = []

    def write_lock(self):
        (self.repo / "cmake/HyleDependencyLock.json").write_text(
            json.dumps(self.lock), encoding="utf-8")

    def tool(self, name="glslangValidator"):
        path = self.directory / (name + (".exe" if os.name == "nt" else ""))
        path.write_text("fixture", encoding="utf-8")
        return path

    def run_success(self, command, **kwargs):
        self.commands.append(command)
        if "--build" in command:
            directory = Path(command[command.index("--build") + 1])
            directory.mkdir(parents=True, exist_ok=True)
            tool = directory / ("glslang.exe" if os.name == "nt" else "glslang")
            tool.write_text("fixture", encoding="utf-8")
            (directory / "glslang-path-Release.txt").write_text(str(tool), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="glslang fixture")

    def test_path_and_explicit_tool_discovery(self):
        tool = self.tool()
        with patch.object(resources.shutil, "which", return_value=str(tool)), \
             patch.object(resources.subprocess, "run", side_effect=self.run_success):
            self.assertEqual(resources.find_glslang(self.environment), tool.resolve())
            self.environment.definitions["HYLE_VULKAN_GLSLANG"] = str(tool)
            self.assertEqual(resources.find_glslang(self.environment), tool.resolve())
        self.environment.definitions["HYLE_VULKAN_GLSLANG"] = str(tool.parent / "missing")
        with self.assertRaisesRegex(RuntimeError, "HYLE_VULKAN_GLSLANG.*usable host tool"):
            resources.find_glslang(self.environment)
        self.environment.definitions["HYLE_VULKAN_GLSLANG"] = str(tool)
        with patch.object(resources.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)):
            with self.assertRaisesRegex(RuntimeError, "usable host tool"):
                resources.find_glslang(self.environment)

    def test_shared_tool_cache_reuses_completed_host_tool(self):
        with patch.object(resources, "find_glslang", return_value=None), \
             patch.object(resources.subprocess, "run", side_effect=self.run_success):
            first = resources.prepare_resources(self.environment, self.cache, offline=False)
            initial_commands = len(self.commands)
            second = resources.prepare_resources(self.environment, self.cache, offline=True)
        self.assertEqual(first, second)
        self.assertEqual(self.commands[initial_commands:], [[first["HYLE_VULKAN_GLSLANG"], "--version"]])
        path = Path(first["HYLE_VULKAN_GLSLANG"])
        self.assertTrue(path.is_relative_to(self.cache.parent / "tools-v1"))
        self.assertFalse(path.is_relative_to(self.cache))

    def test_offline_missing_cache_requests_no_network_and_reports_failure(self):
        with patch.object(resources, "find_glslang", return_value=None), \
             patch.object(resources.subprocess, "run", return_value=subprocess.CompletedProcess([], 1)) as run:
            with self.assertRaisesRegex(RuntimeError, "Offline preparation needs the locked glslang"):
                resources.prepare_resources(self.environment, self.cache, offline=True)
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertIn("-DFETCHCONTENT_FULLY_DISCONNECTED=ON", command)
        self.assertIn("-DFETCHCONTENT_UPDATES_DISCONNECTED=ON", command)
        self.assertNotIn("--build", command)

    def test_lock_and_compiler_changes_isolate_host_build_state(self):
        with patch.object(resources, "find_glslang", return_value=None), \
             patch.object(resources.subprocess, "run", side_effect=self.run_success):
            resources.prepare_resources(self.environment, self.cache, offline=False)
            self.lock["dependencies"][0]["archive"]["sha256"] = "c" * 64
            self.write_lock()
            resources.prepare_resources(self.environment, self.cache, offline=False)
            self.environment.cxx = "another-c++"
            resources.prepare_resources(self.environment, self.cache, offline=False)
        configure = [command for command in self.commands if "-S" in command]
        directories = {command[command.index("-B") + 1] for command in configure}
        self.assertEqual(len(directories), 3)
        self.assertTrue(all(f"-DHYLE_DEP_SOURCE_CACHE={self.cache.as_posix()}" in command
                            for command in configure))

    def test_build_failure_is_reported_with_original_exit_code(self):
        def fail_build(command, **kwargs):
            return subprocess.CompletedProcess(command, 42 if "--build" in command else 0)

        with patch.object(resources, "find_glslang", return_value=None), \
             patch.object(resources.subprocess, "run", side_effect=fail_build):
            with self.assertRaisesRegex(RuntimeError, "preparation failed.*exit 42"):
                resources.prepare_resources(self.environment, self.cache, offline=False)


if __name__ == "__main__":
    unittest.main()
