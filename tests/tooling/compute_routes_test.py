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
          ("CPU", "CUDA", "APPLE", "VULKAN", "ONEDNN", "TORCH_CPU")] + ["HYLE_WITH_ATEN"]
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
              '${_HYLE_COMPUTE_REQUEST_CPU}|${_HYLE_COMPUTE_REQUEST_CUDA}|${_HYLE_COMPUTE_REQUEST_APPLE}|'
              '${_HYLE_COMPUTE_REQUEST_VULKAN}")\n',
            encoding="utf-8")
        result = subprocess.run(["cmake", "-P", str(script)], capture_output=True,
                                text=True, timeout=15, **NO_WINDOW)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return output.read_text() if success else result.stderr

    def test_complete_cpu_default_and_gpu_only_isolation(self):
        self.assertEqual(self.resolve('hyle_resolve_compute_product_routes("graph")'),
                         "cpu|ON|OFF|OFF|OFF")
        self.assertEqual(self.resolve('set(HYLE_COMPUTE_ROUTES cuda)\n'
                                     'hyle_resolve_compute_product_routes("graph")'),
                         "cuda|OFF|ON|OFF|OFF")

    def test_set_canonicalization_is_not_priority(self):
        self.assertEqual(self.resolve('set(HYLE_COMPUTE_ROUTES " CUDA ;cpu;CUDA;vulkan")\n'
                                     'hyle_resolve_compute_product_routes("compute;graph")'),
                         "cpu;cuda;vulkan|ON|ON|OFF|ON")

    def test_invalid_set_and_old_cache_or_parent_variables(self):
        for value in ("", "AUTO", "gpu", "cpu;;cuda", ";cpu", "cpu;"):
            with self.subTest(value=value):
                message = self.resolve(f'set(HYLE_COMPUTE_ROUTES "{value}")\n'
                                       'hyle_resolve_compute_product_routes("graph")', success=False)
                self.assertIn("HYLE_COMPUTE_ROUTES", message)
        for name in LEGACY:
            for kind in ("", ' CACHE STRING ""'):
                for value in (("ON", "OFF") if name == "HYLE_WITH_ATEN" else ("OFF",)):
                    with self.subTest(name=name, kind=kind, value=value):
                        message = self.resolve(f'set({name} {value}{kind})\n'
                                               'hyle_resolve_compute_product_routes("graph")', success=False)
                        self.assertIn(name + " is no longer", message)

    def test_apple_full_cpu_dependencies_and_platform_errors(self):
        self.assertEqual(self.resolve('set(HYLE_COMPUTE_ROUTES "apple;cpu")\n'
                                     'hyle_resolve_compute_product_routes("graph")',
                                     system="Darwin", processor="aarch64"),
                         "cpu;apple|ON|OFF|ON|OFF")
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
                         "|OFF|OFF|OFF|OFF")

    def test_complete_cuda_requires_locked_host_sdk_architecture(self):
        product = ('set(HYLE_COMPUTE_ROUTES cuda)\n'
                   'hyle_resolve_compute_product_routes("graph")')
        for system in ("Windows", "Linux"):
            with self.subTest(system=system):
                self.assertEqual(self.resolve(product, system=system),
                                 "cuda|OFF|ON|OFF|OFF")
                message = self.resolve(product, system=system,
                                       processor="aarch64", success=False)
                self.assertIn("complete CUDA route requires Windows/Linux x86_64", message)
                # Exact backend maintenance must not inherit the full product's
                # locked ATen host restriction before its own SDK is checked.
                self.assertEqual(self.resolve(
                    'set(HYLE_BUILD_COMPUTE_BACKEND_CUDA ON)\n'
                    'hyle_resolve_compute_factor_routes()',
                    system=system, processor="aarch64"),
                    "cuda|OFF|ON|OFF|OFF")

    def test_developer_factor_closure_and_platform_off_headers(self):
        body = ('set(HYLE_BUILD_COMPUTE_BACKEND_CPU ON)\n'
                'set(HYLE_BUILD_COMPUTE_BACKEND_CUDA ON)\n'
                'set(HYLE_BUILD_COMPUTE_ROUTE_APPLE ON)\n'
                'hyle_resolve_compute_factor_routes()')
        self.assertEqual(self.resolve(body), "cpu;cuda|ON|ON|OFF|OFF")
        self.assertEqual(self.resolve(body, system="Darwin", processor="aarch64"),
                         "cpu;apple|ON|OFF|ON|OFF")
        self.resolve('set(HYLE_COMPUTE_ROUTES cpu)\n'
                     'hyle_resolve_compute_factor_routes()', success=False)

    def test_actual_product_dag_prunes_unselected_native_roots(self):
        # Actual manifest/configure path, no SDK stubs and no provider mock.
        for routes, expected, absent in (
            ("cpu", "compute_route_cpu;graph_route_cpu;compute_provider_native",
             "compute_backend_cuda;compute_route_vulkan;compute_route_apple;graph_route_cpu_cuda"),
            ("cuda", "compute_backend_cuda;compute_route_cuda;graph_route_cuda",
             "compute_backend_cpu;compute_provider_native;compute_route_vulkan;graph_route_cpu_cuda"),
            ("vulkan", "compute_route_vulkan;graph_route_vulkan;compute_provider_iree_vulkan",
             "compute_backend_cpu;compute_backend_cuda;compute_route_apple;graph_route_cpu_cuda"),
            ("cpu;cuda", "graph_route_cpu_cuda;graph_route_cpu;graph_route_cuda;compute_provider_onednn",
             "compute_route_vulkan;compute_route_apple;graph_progress"),
            ("cpu;vulkan", "graph_route_cpu;graph_route_vulkan",
             "compute_backend_cuda;compute_route_apple;graph_route_cpu_cuda"),
        ):
            with self.subTest(routes=routes):
                source = self.directory / routes.replace(";", "-")
                source.mkdir()
                checks = ""
                for factor in expected.split(";"):
                    checks += f'if(NOT HYLE_BUILD_{factor.upper()})\nmessage(FATAL_ERROR "missing {factor}")\nendif()\n'
                for factor in (absent + ";compute_stream;io_core;sched").split(";"):
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

    def test_file_stream_is_an_explicit_route_scoped_service(self):
        # The real product resolver only; no SDK, providers or generated targets.
        cases = (
            ("compute", "cpu", "Windows", "x86_64",
             "compute_route_cpu", "compute_stream;io_core;sched"),
            ("graph;compute_stream", "vulkan", "Windows", "x86_64",
             "graph_route_vulkan;compute_stream;compute_stream_vulkan;io_core",
             "compute_stream_cpu;compute_stream_cuda;compute_stream_apple;compute_backend_cpu;io_tls;io_http;graph_progress"),
            ("compute_stream", "cpu", "Linux", "aarch64",
             "compute_stream;compute_stream_cpu;compute_backend_cpu;io_core",
             "compute_route_cpu;compute_provider_native;compute_provider_onednn;compute_provider_ggml;graph;io_tls;io_http"),
        )
        for ordinal, (products, routes, system, processor, expected, absent) in enumerate(cases):
            with self.subTest(products=products, routes=routes):
                source = self.directory / f"stream-{ordinal}"
                source.mkdir()
                checks = ""
                for factor in expected.split(";"):
                    checks += f'if(NOT HYLE_BUILD_{factor.upper()})\nmessage(FATAL_ERROR "missing {factor}")\nendif()\n'
                for factor in absent.split(";"):
                    checks += f'if(HYLE_BUILD_{factor.upper()})\nmessage(FATAL_ERROR "unselected {factor}")\nendif()\n'
                (source / "CMakeLists.txt").write_text(
                    'cmake_minimum_required(VERSION 3.25)\nproject(stream_metadata LANGUAGES NONE)\n'
                    f'set(CMAKE_SYSTEM_NAME "{system}")\nset(HYLE_TARGET_PROCESSOR "{processor}")\n'
                    f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
                    'set(HYLE_FCONTEXT_SUPPORTED TRUE)\nset(HYLE_BUILD_TESTS OFF)\n'
                    f'set(HYLE_COMPONENTS "{products}")\nset(HYLE_COMPUTE_ROUTES "{routes}")\n'
                    f'include("{(ROOT / "cmake/HyleComponents.cmake").as_posix()}")\n'
                    + checks, encoding="utf-8")
                result = subprocess.run(["cmake", "-S", str(source), "-B", str(source / "build")],
                                        capture_output=True, text=True, timeout=20, **NO_WINDOW)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_graph_progress_is_an_explicit_route_neutral_service(self):
        for ordinal, (products, enabled) in enumerate((
            ("graph;sched", False), ("graph;graph_progress", True),
            ("graph_progress", True),
        )):
            with self.subTest(products=products):
                source = self.directory / f"progress-{ordinal}"
                source.mkdir()
                checks = f'''
if(NOT HYLE_BUILD_SCHED OR NOT HYLE_BUILD_GRAPH)
  message(FATAL_ERROR "missing service dependencies")
endif()
if(NOT HYLE_BUILD_GRAPH_PROGRESS STREQUAL "{'TRUE' if enabled else 'FALSE'}")
  message(FATAL_ERROR "progress was not selected explicitly")
endif()
if("graph_progress" IN_LIST _HYLE_PRODUCT_FACTORS_GRAPH)
  message(FATAL_ERROR "progress polluted the ordinary Graph facade")
endif()
'''
                if products == "graph_progress":
                    checks += '''
if(HYLE_COMPUTE_ROUTE_LIST OR HYLE_BUILD_GRAPH_BATCH OR
   HYLE_BUILD_COMPUTE_ROUTE_CPU OR HYLE_BUILD_COMPUTE_ROUTE_VULKAN OR
   HYLE_BUILD_COMPUTE_PROVIDER_NATIVE OR HYLE_BUILD_IO_CORE)
  message(FATAL_ERROR "standalone progress acquired a default route or IO")
endif()
'''
                (source / "CMakeLists.txt").write_text(
                    'cmake_minimum_required(VERSION 3.25)\nproject(progress_metadata LANGUAGES NONE)\n'
                    f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
                    'set(HYLE_TARGET_PROCESSOR x86_64)\nset(HYLE_FCONTEXT_SUPPORTED TRUE)\n'
                    'set(HYLE_BUILD_TESTS OFF)\nset(HYLE_COMPUTE_ROUTES vulkan)\n'
                    f'set(HYLE_COMPONENTS "{products}")\n'
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

    def test_apple_composition_uses_requested_endpoints_without_progress(self):
        # Factor selection only. Native Apple build/runtime use the actual Air.
        for routes, expected in (("apple", False), ("cpu;apple", True)):
            with self.subTest(routes=routes):
                source = self.directory / routes.replace(";", "-")
                source.mkdir()
                (source / "CMakeLists.txt").write_text(
                    'cmake_minimum_required(VERSION 3.25)\nproject(apple_metadata LANGUAGES NONE)\n'
                    'set(CMAKE_SYSTEM_NAME Darwin)\nset(HYLE_TARGET_PROCESSOR aarch64)\n'
                    'set(HYLE_FCONTEXT_SUPPORTED TRUE)\n'
                    f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
                    'set(HYLE_COMPONENTS graph)\nset(HYLE_BUILD_TESTS OFF)\n'
                    f'set(HYLE_COMPUTE_ROUTES "{routes}")\n'
                    f'include("{(ROOT / "cmake/HyleComponents.cmake").as_posix()}")\n'
                    'if(NOT HYLE_BUILD_GRAPH_ROUTE_APPLE OR HYLE_BUILD_GRAPH_PROGRESS)\n'
                    'message(FATAL_ERROR "Apple graph endpoint or optional progress differs")\nendif()\n'
                    f'if({"NOT " if expected else ""}HYLE_BUILD_GRAPH_ROUTE_CPU)\n'
                    'message(FATAL_ERROR "CPU endpoint differs from requested routes")\nendif()\n',
                    encoding="utf-8")
                result = subprocess.run(["cmake", "-S", str(source), "-B", str(source / "build")],
                                        capture_output=True, text=True, timeout=20, **NO_WINDOW)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_installed_requirements_assert_sdk_without_reconfiguring(self):
        # Package metadata fixture, not a native library/provider certificate.
        prepare = self.directory / "prepare"
        sdk = self.directory / "sdk/lib/cmake/Hyle"
        prepare.mkdir()
        (prepare / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\nproject(SdkMetadata NONE)\n'
            'include(CMakePackageConfigHelpers)\n'
            'set(HYLE_PACKAGE_COMPUTE_ROUTES cpu)\n'
            'set(HYLE_PACKAGE_COMPUTE_RECIPE_ID fixture-recipe)\n'
            'set(HYLE_PACKAGE_COMPONENTS compute)\n'
            'set(HYLE_PACKAGE_FEATURES compute_cpu)\n'
            'set(HYLE_PACKAGE_STATIC ON)\n'
            f'file(MAKE_DIRECTORY "{sdk.as_posix()}")\n'
            f'configure_package_config_file("{(ROOT / "cmake/HyleConfig.cmake.in").as_posix()}" '
            f'"{(sdk / "HyleConfig.cmake").as_posix()}" INSTALL_DESTINATION lib/cmake/Hyle)\n'
            f'file(WRITE "{(sdk / "HyleTargets-product_compute.cmake").as_posix()}" '
            '"add_library(hyle::compute INTERFACE IMPORTED)\\n")\n'
            f'file(WRITE "{(sdk / "HylePackageIndex.cmake").as_posix()}" '
            '"set(_Hyle_package_group_order product_compute)\\n'
            'set(_Hyle_product_compute_groups product_compute)\\n'
            'set(_Hyle_product_compute_required_groups product_compute)\\n'
            'set(_Hyle_group_product_compute_file HyleTargets-product_compute.cmake)\\n")\n'
            f'file(COPY "{(ROOT / "cmake/HyleRuntimeStaging.cmake").as_posix()}" '
            f'DESTINATION "{sdk.as_posix()}")\n', encoding="utf-8")
        result = subprocess.run(["cmake", "-S", str(prepare), "-B", str(prepare / "build")],
                                capture_output=True, text=True, timeout=15, **NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        source = self.directory / "consumer"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\nproject(ConsumeMetadata NONE)\n'
            'find_package(Hyle CONFIG REQUIRED COMPONENTS compute)\n'
            'if(NOT Hyle_AVAILABLE_COMPUTE_ROUTES STREQUAL "cpu" OR '
            'NOT Hyle_COMPUTE_RECIPE_ID STREQUAL "fixture-recipe")\n'
            'message(FATAL_ERROR "SDK metadata changed")\nendif()\n', encoding="utf-8")
        cases = [([], True), (["-DHYLE_COMPUTE_ROUTES=CPU"], True),
                 (["-DHYLE_COMPUTE_ROUTES=cpu;cuda"], False),
                 (["-DHYLE_COMPUTE_ROUTES="], False),
                 (["-DHYLE_COMPUTE_ROUTES=cpu;;cpu"], False),
                 (["-DHYLE_COMPUTE_TORCH_CPU=OFF"], False),
                 (["-DHYLE_WITH_ATEN=ON"], False),
                 (["-DHYLE_WITH_ATEN=OFF"], False)]
        for ordinal, (args, expected) in enumerate(cases):
            result = subprocess.run(["cmake", "-S", str(source), "-B", str(source / str(ordinal)),
                                     "-DHyle_DIR=" + str(sdk), *args], capture_output=True,
                                    text=True, timeout=15, **NO_WINDOW)
            self.assertEqual(result.returncode == 0, expected, result.stdout + result.stderr)


    def test_delivery_wrapper_preserves_child_route_argument(self):
        driver = (ROOT / "cmake/HyleCiSelectedComponentSmoke.cmake").read_text()
        start = driver.index("function(_hyle_selected_run label)")
        end = driver.index("endfunction()", start) + len("endfunction()")
        wrapper = driver[start:end]
        for routes in ("cpu", "cpu;cuda", "cpu;apple", "cpu;vulkan"):
            with self.subTest(routes=routes):
                check = self.directory / "child.cmake"
                check.write_text(
                    f'set(expected [==[{routes}]==])\n'
                    'if(NOT HYLE_COMPUTE_ROUTES STREQUAL expected)\n'
                    'message(FATAL_ERROR "child route argument was split: ${HYLE_COMPUTE_ROUTES}")\n'
                    'endif()\n', encoding="utf-8")
                script = self.directory / "forward.cmake"
                script.write_text(
                    'cmake_minimum_required(VERSION 3.25)\n' + wrapper + '\n'
                    f'set(routes [==[{routes}]==])\n'
                    'string(REPLACE ";" "\\\\;" route_argument "${routes}")\n'
                    'list(APPEND child_args "-DHYLE_COMPUTE_ROUTES=${route_argument}")\n'
                    f'_hyle_selected_run(probe "${{CMAKE_COMMAND}}" ${{child_args}} -P "{check.as_posix()}")\n',
                    encoding="utf-8")
                result = subprocess.run(["cmake", "-P", str(script)], capture_output=True,
                                        text=True, timeout=15, **NO_WINDOW)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


    def test_delivery_guard_requires_the_complete_route_set(self):
        driver = (ROOT / "cmake/HyleCiSelectedComponentSmoke.cmake").read_text()
        start = driver.index("function(_hyle_selected_require_compute_routes config routes)")
        end = driver.index("endfunction()", start) + len("endfunction()")
        guard = driver[start:end]
        cases = [(value, value, True) for value in
                 ("cpu", "cuda", "apple", "vulkan", "cpu;cuda", "cpu;apple", "cpu;vulkan")]
        cases += [("cpu", "cpu;cuda", False), ("cpu", "cpu;apple", False),
                  ("cpu", "cpu;vulkan", False), ("cpu;cuda", "cpu", False),
                  (None, "cpu", False)]
        for actual, expected, success in cases:
            with self.subTest(actual=actual, expected=expected):
                config = '' if actual is None else f'set(Hyle_AVAILABLE_COMPUTE_ROUTES "{actual}")'
                script = self.directory / "sdk-route-check.cmake"
                script.write_text(
                    'cmake_minimum_required(VERSION 3.25)\n' + guard + '\n'
                    f'_hyle_selected_require_compute_routes([==[{config}]==] [==[{expected}]==])\n',
                    encoding="utf-8")
                result = subprocess.run(["cmake", "-P", str(script)], capture_output=True,
                                        text=True, timeout=15, **NO_WINDOW)
                self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
                if not success:
                    self.assertIn("do not match requested", result.stderr)


if __name__ == "__main__":
    unittest.main()
