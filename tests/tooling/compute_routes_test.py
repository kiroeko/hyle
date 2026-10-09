"""Configure/metadata contracts only; this does not qualify native execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ["HYLE_COMPUTE_" + item for item in
          ("CPU", "CUDA", "APPLE", "VULKAN", "ONEDNN", "TORCH_CPU")]
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


class ComputeRoutesTest(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / ".scratch/validation/compute-routes-contract"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.directory = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def resolve(self, body, *, system="Windows", processor="x86_64", success=True):
        script = self.directory / "resolve.cmake"
        output = self.directory / "result.txt"
        script.write_text(
            'cmake_minimum_required(VERSION 3.25)\n'
            f'set(CMAKE_SYSTEM_NAME "{system}")\n'
            f'set(HYLE_TARGET_PROCESSOR "{processor}")\n'
            f'include("{(ROOT / "cmake/HyleComputeRoutes.cmake").as_posix()}")\n'
            + body + '\n'
            + f'file(WRITE "{output.as_posix()}" "${{HYLE_COMPUTE_ROUTE_LIST}}|'
              '${HYLE_COMPUTE_CPU}|${HYLE_COMPUTE_CUDA}|${HYLE_COMPUTE_APPLE}|'
              '${HYLE_COMPUTE_VULKAN}|${HYLE_COMPUTE_ONEDNN}|${HYLE_COMPUTE_TORCH_CPU}")\n',
            encoding="utf-8")
        result = subprocess.run(["cmake", "-P", str(script)], capture_output=True,
                                text=True, timeout=15, **NO_WINDOW)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return output.read_text() if success else result.stderr

    def test_complete_cpu_default_and_gpu_only_isolation(self):
        self.assertEqual(self.resolve('hyle_resolve_compute_product_routes("graph")'),
                         "cpu|ON|OFF|OFF|OFF|ON|ON")
        self.assertEqual(self.resolve('set(HYLE_COMPUTE_ROUTES cuda)\n'
                                     'hyle_resolve_compute_product_routes("graph")'),
                         "cuda|OFF|ON|OFF|OFF|OFF|OFF")

    def test_set_canonicalization_is_not_priority(self):
        self.assertEqual(self.resolve('set(HYLE_COMPUTE_ROUTES " CUDA ;cpu;CUDA;vulkan")\n'
                                     'hyle_resolve_compute_product_routes("compute;graph")'),
                         "cpu;cuda;vulkan|ON|ON|OFF|ON|ON|ON")

    def test_invalid_set_and_old_cache_or_parent_variables(self):
        for value in ("", "AUTO", "gpu", "cpu;;cuda", ";cpu", "cpu;"):
            with self.subTest(value=value):
                message = self.resolve(f'set(HYLE_COMPUTE_ROUTES "{value}")\n'
                                       'hyle_resolve_compute_product_routes("graph")', success=False)
                self.assertIn("HYLE_COMPUTE_ROUTES", message)
        for name in LEGACY:
            for kind in ("", ' CACHE STRING ""'):
                with self.subTest(name=name, kind=kind):
                    message = self.resolve(f'set({name} OFF{kind})\n'
                                           'hyle_resolve_compute_product_routes("graph")', success=False)
                    self.assertIn(name + " is no longer", message)

    def test_apple_full_cpu_dependencies_and_platform_errors(self):
        self.assertEqual(self.resolve('set(HYLE_COMPUTE_ROUTES "apple;cpu")\n'
                                     'hyle_resolve_compute_product_routes("graph")',
                                     system="Darwin", processor="aarch64"),
                         "cpu;apple|ON|OFF|ON|OFF|OFF|ON")
        self.resolve('set(HYLE_COMPUTE_ROUTES apple)\n'
                     'hyle_resolve_compute_product_routes("graph")', success=False)
        self.resolve('set(HYLE_COMPUTE_ROUTES cuda)\n'
                     'hyle_resolve_compute_product_routes("graph")',
                     system="Darwin", processor="aarch64", success=False)
        message = self.resolve('hyle_resolve_compute_product_routes("compute")',
                               system="Linux", processor="aarch64", success=False)
        self.assertIn("complete CPU", message)

    def test_unrelated_products_do_not_acquire_compute(self):
        self.assertEqual(self.resolve('hyle_resolve_compute_product_routes("core;mem")'),
                         "|OFF|OFF|OFF|OFF|OFF|OFF")

    def test_developer_factor_closure_and_platform_off_headers(self):
        body = ('set(HYLE_BUILD_COMPUTE_BACKEND_CPU ON)\n'
                'set(HYLE_BUILD_COMPUTE_BACKEND_CUDA ON)\n'
                'set(HYLE_BUILD_COMPUTE_ROUTE_APPLE ON)\n'
                'hyle_resolve_compute_factor_routes()')
        self.assertEqual(self.resolve(body), "cpu;cuda|ON|ON|OFF|OFF|ON|ON")
        self.assertEqual(self.resolve(body, system="Darwin", processor="aarch64"),
                         "cpu;apple|ON|OFF|ON|OFF|OFF|ON")
        self.resolve('set(HYLE_COMPUTE_ROUTES cpu)\n'
                     'hyle_resolve_compute_factor_routes()', success=False)

    def test_actual_product_dag_prunes_unselected_native_roots(self):
        # Actual manifest/configure path, no SDK stubs and no provider mock.
        for routes, expected, absent in (
            ("cpu", "compute_route_cpu;graph_route_cpu;compute_provider_native",
             "compute_backend_cuda;compute_route_vulkan;compute_route_apple"),
            ("cuda", "compute_backend_cuda;compute_route_cuda;graph_route_cuda",
             "compute_backend_cpu;compute_provider_native;compute_route_vulkan"),
            ("vulkan", "compute_route_vulkan;graph_route_vulkan;compute_provider_iree_vulkan",
             "compute_backend_cpu;compute_backend_cuda;compute_route_apple"),
        ):
            with self.subTest(routes=routes):
                source = self.directory / routes
                source.mkdir()
                checks = ""
                for factor in expected.split(";"):
                    checks += f'if(NOT HYLE_BUILD_{factor.upper()})\nmessage(FATAL_ERROR "missing {factor}")\nendif()\n'
                for factor in absent.split(";"):
                    checks += f'if(HYLE_BUILD_{factor.upper()})\nmessage(FATAL_ERROR "unselected {factor}")\nendif()\n'
                (source / "CMakeLists.txt").write_text(
                    'cmake_minimum_required(VERSION 3.25)\nproject(route_metadata LANGUAGES NONE)\n'
                    f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
                    'set(HYLE_TARGET_PROCESSOR x86_64)\nset(HYLE_FCONTEXT_SUPPORTED TRUE)\n'
                    'set(HYLE_COMPONENTS graph)\nset(HYLE_BUILD_TESTS OFF)\n'
                    f'set(HYLE_COMPUTE_ROUTES "{routes}")\n'
                    f'include("{(ROOT / "cmake/HyleComponents.cmake").as_posix()}")\n'
                    + checks, encoding="utf-8")
                result = subprocess.run(["cmake", "-S", str(source), "-B", str(source / "build")],
                                        capture_output=True, text=True, timeout=20, **NO_WINDOW)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_four_presets_share_one_input_and_bounded_parallelism(self):
        presets = json.loads((ROOT / "CMakePresets.json").read_text())
        configure = {row["name"]: row for row in presets["configurePresets"]}
        builds = {row["name"]: row for row in presets["buildPresets"]}
        for route in ("cpu", "cuda", "apple", "vulkan"):
            name = "compute-" + route
            cache = configure[name]["cacheVariables"]
            self.assertEqual(cache["HYLE_COMPUTE_ROUTES"], route)
            self.assertFalse(set(cache) & set(LEGACY))
            self.assertEqual(builds[name]["jobs"], 2)


if __name__ == "__main__":
    unittest.main()
