"""Resolve application/library build roots without configuring native SDKs."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest

if __package__:
    from .inference_architecture_boundaries_test import ArchitectureBoundariesTest
else:
    from inference_architecture_boundaries_test import ArchitectureBoundariesTest


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".scratch/validation/engine-components"
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


class EngineComponentsTest(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def resolve(self, name, selection=None, *, embedded=False, routes="cpu",
                checks="", system="Windows", processor="x86_64", success=True):
        source = self.directory / name
        source.mkdir()
        body = (
            'cmake_minimum_required(VERSION 3.25)\n'
            'project(EngineSelection LANGUAGES NONE)\n'
            f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
            f'set(CMAKE_SYSTEM_NAME "{system}")\n'
            f'set(HYLE_TARGET_PROCESSOR "{processor}")\n'
            'set(HYLE_FCONTEXT_SUPPORTED TRUE)\n'
            'set(HYLE_BUILD_TESTS OFF)\n'
        )
        if embedded:
            body += 'set(PROJECT_IS_TOP_LEVEL FALSE)\n'
        if selection is not None:
            body += f'set(HYLE_COMPONENTS "{selection}")\n'
        if routes is not None:
            body += f'set(HYLE_COMPUTE_ROUTES "{routes}")\n'
        body += (
            'include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")\n'
            'if("engine" IN_LIST HYLE_KNOWN_PRODUCTS OR\n'
            '   "engine" IN_LIST HYLE_RESOLVED_PRODUCT_LIST OR\n'
            '   "engine" IN_LIST HYLE_PRODUCTION_PRODUCT_LIST)\n'
            '  message(FATAL_ERROR "application leaked into C++ SDK products")\n'
            'endif()\n' + checks
        )
        (source / "CMakeLists.txt").write_text(body, encoding="utf-8")
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(source / "build")],
            capture_output=True, text=True, timeout=20, **NO_WINDOW)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return result

    @staticmethod
    def engine_checks():
        return '''
if(NOT HYLE_BUILD_ENGINE)
  message(FATAL_ERROR "engine application root was lost")
endif()
foreach(product IN ITEMS compute graph graph_progress compute_stream mem)
  if(NOT product IN_LIST HYLE_PRODUCTION_PRODUCT_LIST)
    message(FATAL_ERROR "engine missing ${product} library facade")
  endif()
endforeach()
foreach(factor IN ITEMS compute_state compute_batch graph_batch graph_progress compute_stream)
  string(TOUPPER "${factor}" upper)
  if(NOT HYLE_BUILD_${upper})
    message(FATAL_ERROR "engine missing ${factor} implementation")
  endif()
endforeach()
'''

    def test_standalone_default_is_engine(self):
        self.resolve("default", checks=self.engine_checks() + '''
if(NOT HYLE_COMPONENTS STREQUAL "engine")
  message(FATAL_ERROR "top-level default is not engine")
endif()
''')

    def test_explicit_engine_uses_existing_route_closures(self):
        for route, system, processor in (
            ("cpu", "Windows", "x86_64"),
            ("cuda", "Linux", "x86_64"),
            ("apple", "Darwin", "aarch64"),
            ("vulkan", "Windows", "x86_64"),
            ("cpu;cuda", "Windows", "x86_64"),
        ):
            with self.subTest(route=route):
                self.resolve(route.replace(";", "-"), "engine", routes=route,
                             system=system, processor=processor,
                             checks=self.engine_checks() + f'''
if(NOT HYLE_COMPUTE_ROUTE_LIST STREQUAL "{route}")
  message(FATAL_ERROR "engine changed selected route identity")
endif()
''')

    def test_util_stays_independent_and_does_not_enable_compute(self):
        for embedded in (False, True):
            with self.subTest(embedded=embedded):
                self.resolve("util-" + str(embedded), "util", embedded=embedded,
                             routes=None, checks='''
if(HYLE_BUILD_ENGINE OR HYLE_BUILD_COMPUTE OR HYLE_BUILD_GRAPH OR
   HYLE_BUILD_COMPUTE_STREAM OR HYLE_BUILD_GRAPH_PROGRESS OR
   HYLE_COMPUTE_ROUTE_LIST OR DEFINED HYLE_COMPUTE_ROUTES)
  message(FATAL_ERROR "util unexpectedly acquired inference dependencies")
endif()
if(NOT HYLE_BUILD_PRODUCT_UTIL)
  message(FATAL_ERROR "util library facade disappeared")
endif()
''')

    def test_compute_and_graph_roots_do_not_enable_engine(self):
        # The application is a consumer, not an implicit dependency of either
        # reusable facade, including when both are selected together.
        for selection in ("compute", "graph", "compute;graph"):
            with self.subTest(selection=selection):
                expected = "".join(
                    f'if(NOT HYLE_BUILD_PRODUCT_{product.upper()})\n'
                    f'  message(FATAL_ERROR "selected {product} facade is missing")\nendif()\n'
                    for product in selection.split(";")
                )
                self.resolve("library-" + selection.replace(";", "-"), selection,
                             routes="cuda", checks=expected + '''
if(HYLE_BUILD_ENGINE)
  message(FATAL_ERROR "a library-only selection unexpectedly enabled the engine")
endif()
''')

    def test_all_has_application_only_for_standalone_build(self):
        self.resolve("all", "all", checks=self.engine_checks())
        self.resolve("embedded-all", "all", embedded=True, checks='''
if(HYLE_BUILD_ENGINE)
  message(FATAL_ERROR "embedding all libraries gained an application")
endif()
''')

    def test_embedding_can_explicitly_request_application(self):
        self.resolve("embedded-engine", "engine;util", embedded=True,
                     checks=self.engine_checks() + '''
if(NOT HYLE_BUILD_PRODUCT_UTIL)
  message(FATAL_ERROR "additional explicit library root was lost")
endif()
''')

    def test_actual_application_target_is_default_and_probes_are_opt_in(self):
        # Configure the real application directories against tiny SDK interface
        # targets. This checks source layout and target ownership without
        # fetching native libraries or compiling the model implementation.
        source = self.directory / "application"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\n'
            'project(EngineTargets LANGUAGES CXX)\n'
            f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
            'set(HYLE_COMPUTE_ROUTE_LIST cpu)\n'
            'foreach(product IN ITEMS compute graph graph_progress compute_stream mem)\n'
            '  add_library(sdk_${product} INTERFACE)\n'
            '  add_library(hyle::${product} ALIAS sdk_${product})\n'
            'endforeach()\n'
            'include("${HYLE_SOURCE_DIR}/cmake/HyleRuntimeStaging.cmake")\n'
            'add_subdirectory("${HYLE_SOURCE_DIR}/engine" engine)\n'
            'get_target_property(name hyle_engine OUTPUT_NAME)\n'
            'get_target_property(excluded hyle_engine EXCLUDE_FROM_ALL)\n'
            'if(NOT name STREQUAL "hyle" OR excluded)\n'
            '  message(FATAL_ERROR "engine is not the default Hyle application")\n'
            'endif()\n'
            'get_target_property(diagnostic_excluded hyle_qwen35_diagnostic EXCLUDE_FROM_ALL)\n'
            'get_target_property(product_sources hyle_engine SOURCES)\n'
            'get_target_property(diagnostic_sources hyle_qwen35_diagnostic SOURCES)\n'
            'if(NOT diagnostic_excluded OR NOT "${product_sources}" MATCHES "generation_runtime[.]cc" OR\n'
            '   "${product_sources}" MATCHES "qwen35/main[.]cc" OR NOT "main.cc" IN_LIST diagnostic_sources)\n'
            '  message(FATAL_ERROR "product runtime and explicit diagnostic consumer are not separated")\n'
            'endif()\n'
            'get_target_property(probe_excluded hyle_qwen35_mtp_session_probe EXCLUDE_FROM_ALL)\n'
            'if(NOT probe_excluded)\n'
            '  message(FATAL_ERROR "real-weight diagnostic became a default application")\n'
            'endif()\n'
            'if(TARGET hyle_qwen35_random_mtp_probe)\n'
            '  message(FATAL_ERROR "CPU-only build acquired a native random-MTP probe")\n'
            'endif()\n'
            'if(TARGET hyle_product_engine OR TARGET hyle::engine)\n'
            '  message(FATAL_ERROR "executable application manufactured a C++ SDK facade")\n'
            'endif()\n', encoding="utf-8")
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(source / "build")],
            capture_output=True, text=True, timeout=30, **NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cuda_tests_off_keeps_kimi_diagnostic_out_of_application(self):
        # Exercise actual application CMake targets, including the CUDA branch
        # that registers the private Kimi component. No native SDK is compiled.
        source = self.directory / "cuda-application"
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\n'
            'project(CudaEngineTargets LANGUAGES CXX)\n'
            f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
            'set(HYLE_BUILD_TESTS OFF)\n'
            'set(HYLE_COMPUTE_ROUTE_LIST "cpu;cuda")\n'
            'foreach(product IN ITEMS compute graph graph_progress compute_stream mem)\n'
            '  add_library(sdk_${product} INTERFACE)\n'
            '  add_library(hyle::${product} ALIAS sdk_${product})\n'
            'endforeach()\n'
            'include("${HYLE_SOURCE_DIR}/cmake/HyleRuntimeStaging.cmake")\n'
            'add_subdirectory("${HYLE_SOURCE_DIR}/engine" engine)\n'
            'if(NOT TARGET hyle_qwen35_random_mtp_probe)\n'
            '  message(FATAL_ERROR "CUDA application lost its opt-in sampled MTP probe")\n'
            'endif()\n'
            'get_target_property(random_probe_excluded hyle_qwen35_random_mtp_probe EXCLUDE_FROM_ALL)\n'
            'get_target_property(random_probe_sources hyle_qwen35_random_mtp_probe SOURCES)\n'
            'if(NOT random_probe_excluded OR NOT "${random_probe_sources}" MATCHES "tests/tools/qwen_random_mtp_probe[.]cc" OR NOT "native_gate.cc" IN_LIST random_probe_sources)\n'
            '  message(FATAL_ERROR "random MTP probe is not opt-in or lost its original source/gate")\n'
            'endif()\n'
            'if(TARGET hyle_kimi_kda_mla_probe)\n'
            '  message(FATAL_ERROR "testsOFF application acquired the Kimi test diagnostic")\n'
            'endif()\n'
            'if(NOT TARGET hyle_kimi_k3_component)\n'
            '  message(FATAL_ERROR "CUDA application lost its model-private component")\n'
            'endif()\n'
            'get_target_property(excluded hyle_kimi_k3_component EXCLUDE_FROM_ALL)\n'
            'get_target_property(kind hyle_kimi_k3_component TYPE)\n'
            'get_target_property(links hyle_engine LINK_LIBRARIES)\n'
            'if(NOT excluded OR NOT kind STREQUAL "STATIC_LIBRARY")\n'
            '  message(FATAL_ERROR "Kimi component became a default product target")\n'
            'endif()\n'
            'if("hyle_kimi_k3_component" IN_LIST links)\n'
            '  message(FATAL_ERROR "default Qwen application linked the Kimi component")\n'
            'endif()\n', encoding="utf-8")
        result = subprocess.run(
            ["cmake", "-S", str(source), "-B", str(source / "build")],
            capture_output=True, text=True, timeout=30, **NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_empty_or_ambiguous_selection_is_rejected(self):
        self.resolve("empty", "", success=False)
        self.resolve("all-and-engine", "all;engine", success=False)
        self.resolve("missing-parent-selection", embedded=True, success=False)

    def model_component(self, family):
        # Exercise the private component without the application or test SDK.
        # The same target must also be present but unlinked in the application.
        for application in (False, True):
            source = self.directory / (family + str(application))
            source.mkdir()
            body = (
                'cmake_minimum_required(VERSION 3.25)\n'
                'project(ModelBoundary LANGUAGES CXX)\n'
                f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
                'set(HYLE_COMPUTE_ROUTE_LIST "cpu;cuda")\n'
                'foreach(product IN ITEMS compute graph graph_progress compute_stream mem)\n'
                '  add_library(sdk_${product} INTERFACE)\n'
                '  add_library(hyle::${product} ALIAS sdk_${product})\n'
                'endforeach()\n')
            if application:
                body += (
                    'add_library(hyle_mem_vm INTERFACE)\n'
                    'include("${HYLE_SOURCE_DIR}/cmake/HyleRuntimeStaging.cmake")\n'
                    'add_subdirectory("${HYLE_SOURCE_DIR}/engine" engine)\n'
                    'get_target_property(links hyle_engine LINK_LIBRARIES)\n'
                    f'if("hyle_{family}_component" IN_LIST links)\n'
                    '  message(FATAL_ERROR "default Qwen linked an unrelated model")\n'
                    'endif()\n')
            else:
                body += (
                    f'include("${{HYLE_SOURCE_DIR}}/engine/models/{family}/component_sources.cmake")\n'
                    f'_hyle_add_{family}_component(hyle::graph hyle::compute hyle::mem)\n')
            body += f'''
get_target_property(kind hyle_{family}_component TYPE)
get_target_property(excluded hyle_{family}_component EXCLUDE_FROM_ALL)
get_target_property(links hyle_{family}_component LINK_LIBRARIES)
get_target_property(sources hyle_{family}_component SOURCES)
if(NOT kind STREQUAL "STATIC_LIBRARY" OR NOT excluded)
  message(FATAL_ERROR "model component escaped its private build boundary")
endif()
foreach(dependency IN ITEMS hyle::graph hyle::compute hyle::mem)
  if(NOT dependency IN_LIST links)
    message(FATAL_ERROR "model lost its actual SDK dependency")
  endif()
endforeach()
if(TARGET hyle_mem_vm AND NOT "hyle_mem_vm" IN_LIST links)
  message(FATAL_ERROR "embedded model lost the actual file-mapping dependency")
endif()
foreach(file IN LISTS sources)
  cmake_path(IS_PREFIX HYLE_SOURCE_DIR "${{file}}" NORMALIZE inside)
  if(NOT inside OR NOT file MATCHES "/engine/models/{family}/[^/]+[.]cc$")
    message(FATAL_ERROR "model implementation still comes from tests/another family")
  endif()
endforeach()
'''
            (source / "CMakeLists.txt").write_text(body, encoding="utf-8")
            result = subprocess.run(["cmake", "-S", str(source), "-B", str(source / "build")],
                                    capture_output=True, text=True, timeout=30, **NO_WINDOW)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_glm_private_component_and_sdk_dependencies(self):
        self.model_component("glm")

    def test_deepseek_private_component_and_sdk_dependencies(self):
        self.model_component("deepseek")

    def test_retirement_gate_follows_the_selected_route_on_macos(self):
        # This source-selection fixture deliberately keeps the host C++
        # compiler. Enabling Objective-C++ is recorded, not executed: no SDK
        # or Apple compiler is required just to check the cold route branch.
        for routes, expected, objective in (
                ("vulkan", "native_gate.cc", "FALSE"),
                ("cpu;vulkan", "native_gate.cc", "FALSE"),
                ("apple;vulkan", "native_gate_apple.mm", "TRUE")):
            with self.subTest(routes=routes):
                source = self.directory / ("gate-" + routes.replace(";", "-"))
                source.mkdir()
                (source / "CMakeLists.txt").write_text(
                    'cmake_minimum_required(VERSION 3.25)\n'
                    'project(GateSelection LANGUAGES CXX)\n'
                    f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n'
                    f'set(HYLE_COMPUTE_ROUTE_LIST "{routes}")\n'
                    'set(APPLE TRUE)\n'
                    'function(enable_language language)\n'
                    '  if(NOT language STREQUAL "OBJCXX")\n'
                    '    message(FATAL_ERROR "unexpected language request")\n'
                    '  endif()\n'
                    '  set(gate_objective TRUE CACHE INTERNAL "fixture")\n'
                    'endfunction()\n'
                    'set(gate_objective FALSE CACHE INTERNAL "fixture")\n'
                    'foreach(product IN ITEMS compute graph graph_progress compute_stream mem)\n'
                    '  add_library(sdk_${product} INTERFACE)\n'
                    '  add_library(hyle::${product} ALIAS sdk_${product})\n'
                    'endforeach()\n'
                    'include("${HYLE_SOURCE_DIR}/cmake/HyleRuntimeStaging.cmake")\n'
                    'add_subdirectory("${HYLE_SOURCE_DIR}/engine" engine)\n'
                    'get_target_property(gate_sources hyle_qwen35_retirement_probe SOURCES)\n'
                    f'if(NOT "{expected}" IN_LIST gate_sources OR NOT gate_objective STREQUAL "{objective}")\n'
                    '  message(FATAL_ERROR "retirement gate differs from selected native route")\n'
                    'endif()\n', encoding="utf-8")
                result = subprocess.run(
                    ["cmake", "-S", str(source), "-B", str(source / "build")],
                    capture_output=True, text=True, timeout=30, **NO_WINDOW)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
