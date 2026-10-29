"""Install/import graph checks with tiny interface targets; no native SDK build."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}

PRODUCER = r'''
cmake_minimum_required(VERSION 3.25)
project(HylePackageFixture VERSION 0.1.0 LANGUAGES NONE)
set(HYLE_SOURCE_DIR "@ROOT@")
set(HYLE_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")
set(HYLE_INSTALL ON)
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_INCLUDEDIR include)
set(CMAKE_INSTALL_BINDIR bin)
set(CMAKE_INSTALL_DATADIR share)
set(HYLE_INSTALL_INCLUDEDIR include)
set(HYLE_COMPUTE_ROUTE_LIST cpu)
set(HYLE_KNOWN_COMPONENTS core mem compute compute_backend_cpu graph sched graph_progress fixture_only)
set(HYLE_KNOWN_PRODUCTS core mem compute graph sched graph_progress fixture_only)
set(HYLE_PRODUCTION_COMPONENT_LIST core mem compute compute_backend_cpu graph sched graph_progress)
set(HYLE_PRODUCTION_PRODUCT_LIST core mem compute graph sched graph_progress)

foreach(_foundation IN ITEMS build_contract trace testing)
  add_library(hyle_${_foundation} INTERFACE)
endforeach()
set_target_properties(hyle_build_contract PROPERTIES EXPORT_NAME _build_contract)
set_target_properties(hyle_trace PROPERTIES EXPORT_NAME _trace_config)
set_target_properties(hyle_testing PROPERTIES EXPORT_NAME _testing_config)
foreach(_factor IN LISTS HYLE_KNOWN_COMPONENTS)
  add_library(hyle_${_factor} INTERFACE)
  set_target_properties(hyle_${_factor} PROPERTIES EXPORT_NAME _factor_${_factor})
  target_link_libraries(hyle_${_factor} INTERFACE hyle_build_contract)
endforeach()
target_link_libraries(hyle_mem INTERFACE hyle_core)
target_link_libraries(hyle_compute INTERFACE hyle_core)
target_link_libraries(hyle_graph INTERFACE hyle_compute)
target_link_libraries(hyle_graph_progress INTERFACE hyle_graph hyle_sched)
add_library(PackageSchedulerSDK::runtime INTERFACE IMPORTED GLOBAL)
target_link_libraries(hyle_sched INTERFACE PackageSchedulerSDK::runtime)
set_target_properties(hyle_compute PROPERTIES HYLE_COMPUTE_CAPABILITIES ggml_cpu)
if(HYLE_STATIC)
  target_compile_definitions(hyle_core INTERFACE HYLE_STATIC)
endif()

add_library(PackageCpuSDK::runtime INTERFACE IMPORTED GLOBAL)
add_library(fake_cpu_vendor INTERFACE)
set_target_properties(fake_cpu_vendor PROPERTIES EXPORT_NAME _fake_cpu_vendor)
target_link_libraries(fake_cpu_vendor INTERFACE PackageCpuSDK::runtime)
target_link_libraries(hyle_compute_backend_cpu INTERFACE
  hyle_compute fake_cpu_vendor)
set_target_properties(hyle_compute_backend_cpu PROPERTIES
  HYLE_COMPUTE_CAPABILITIES cpu)
if(HYLE_STATIC)
  add_library(PackageStaticSDK::runtime INTERFACE IMPORTED GLOBAL)
  target_link_libraries(fake_cpu_vendor INTERFACE PackageStaticSDK::runtime)
endif()
add_library(fake_fixture_vendor INTERFACE)
set_target_properties(fake_fixture_vendor PROPERTIES EXPORT_NAME _fake_fixture_vendor)
add_library(PackageTraceSDK::runtime INTERFACE IMPORTED GLOBAL)
add_library(fake_trace_vendor INTERFACE)
set_target_properties(fake_trace_vendor PROPERTIES EXPORT_NAME _fake_trace_vendor)
target_link_libraries(fake_trace_vendor INTERFACE PackageTraceSDK::runtime)
target_link_libraries(hyle_trace INTERFACE fake_trace_vendor)
# Only a test fixture reaches Trace unless this SDK explicitly ships tracing.
target_link_libraries(hyle_fixture_only INTERFACE hyle_trace)
if(SHIP_TRACE)
  target_link_libraries(hyle_mem INTERFACE hyle_trace)
endif()
# Nested build-only expressions must not make the fixture part of the SDK.
target_link_libraries(hyle_mem INTERFACE
  "$<LINK_ONLY:$<BUILD_INTERFACE:$<IF:$<BOOL:1>,hyle_fixture_only,hyle_trace>>>")

foreach(_product IN LISTS HYLE_KNOWN_PRODUCTS)
  add_library(hyle_product_${_product} INTERFACE)
  set_target_properties(hyle_product_${_product} PROPERTIES EXPORT_NAME ${_product})
  add_library(hyle::${_product} ALIAS hyle_product_${_product})
  target_link_libraries(hyle_product_${_product} INTERFACE hyle_${_product})
endforeach()
target_link_libraries(hyle_product_compute INTERFACE hyle_compute_backend_cpu)

include("${HYLE_SOURCE_DIR}/cmake/HylePackageRegistry.cmake")
set(HYLE_DEPENDENCY_NAMES fixture_runtime)
set(HYLE_DEP_FIXTURE_RUNTIME_CLASS in_tree)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fixture-LICENSE" "fixture source license\n")
string(SHA256 _fixture_license_sha256 "fixture source license\n")
set(HYLE_DEP_FIXTURE_RUNTIME_LICENSE_FILE_SPECS
  "hyle|third_party/fixture/LICENSE|${_fixture_license_sha256}")
add_library(HyleFixtureRuntime::runtime INTERFACE IMPORTED)
target_link_libraries(hyle_core INTERFACE HyleFixtureRuntime::runtime)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/HyleFixtureRuntimeConfig.cmake"
  "if(NOT TARGET HyleFixtureRuntime::runtime)\n"
  "  add_library(HyleFixtureRuntime::runtime INTERFACE IMPORTED)\n"
  "  set_property(TARGET HyleFixtureRuntime::runtime PROPERTY HYLE_FIXTURE_ORIGIN \"\${CMAKE_CURRENT_LIST_DIR}\")\n"
  "endif()\nset(HyleFixtureRuntime_FOUND TRUE)\n")
hyle_register_package_dependency(NAME bundled_runtime OWNER core
  FIND_DEPENDENCIES HyleFixtureRuntime
  LOCKED_DEPENDENCIES fixture_runtime
  RUNTIME_FILE_SPECS
    "${CMAKE_CURRENT_BINARY_DIR}/fixture-LICENSE|share/licenses/Hyle|fixture-LICENSE"
  DEVELOPMENT_FILE_SPECS
    "${CMAKE_CURRENT_BINARY_DIR}/HyleFixtureRuntimeConfig.cmake|lib/cmake/HyleFixtureRuntime")
hyle_register_package_dependency(NAME cpu OWNER compute_backend_cpu
  EXPORT_TARGETS fake_cpu_vendor FIND_DEPENDENCIES PackageCpuSDK
  STATIC_FIND_DEPENDENCIES PackageStaticSDK)
hyle_register_package_dependency(NAME scheduler OWNER sched
  FIND_DEPENDENCIES PackageSchedulerSDK)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/fixture-only.txt" "must not ship\n")
hyle_register_package_dependency(NAME fixture OWNER fixture_only
  FEATURES fixture_only EXPORT_TARGETS fake_fixture_vendor
  FIND_DEPENDENCIES PackageFixtureSDK
  DEVELOPMENT_FILE_SPECS
    "${CMAKE_CURRENT_BINARY_DIR}/fixture-only.txt|share/fixture-only")
hyle_register_package_dependency(NAME trace OWNER foundation
  FEATURES trace EXPORT_TARGETS fake_trace_vendor
  FIND_DEPENDENCIES PackageTraceSDK)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/shared-header.txt" "production header asset\n")
hyle_register_package_dependency(NAME shared_headers OWNER foundation
  CONSUMER_COMPONENTS mem fixture_only FIND_DEPENDENCIES PackageHeadersSDK
  DEVELOPMENT_FILE_SPECS
    "${CMAKE_CURRENT_BINARY_DIR}/shared-header.txt|share/shared-headers")
hyle_register_package_dependency(NAME fixture_headers OWNER foundation
  CONSUMER_COMPONENTS fixture_only FIND_DEPENDENCIES PackageFixtureHeaderSDK
  DEVELOPMENT_FILE_SPECS
    "${CMAKE_CURRENT_BINARY_DIR}/fixture-only.txt|share/fixture-headers")
function(_hyle_component_public_headers factor output)
  if(factor STREQUAL "fixture_only")
    set(${output} "${HYLE_SOURCE_DIR}/include/hyle/util/function_ref.h" PARENT_SCOPE)
  else()
    set(${output} "${HYLE_SOURCE_DIR}/include/hyle/version.h" PARENT_SCOPE)
  endif()
endfunction()
include("${HYLE_SOURCE_DIR}/src/compute_provider_native/torch_cpu_fetch.cmake")
set(_torch_notice_source "${CMAKE_CURRENT_BINARY_DIR}/torch-notice-source")
set(_torch_notice_root "${_torch_notice_source}/third_party/torch_sdk")
foreach(_notice IN ITEMS LICENSE NOTICE ORIGIN.json)
  file(WRITE "${_torch_notice_root}/${_notice}" "fixture Torch ${_notice}\n")
endforeach()
foreach(_platform IN ITEMS windows linux macos)
  file(WRITE "${_torch_notice_root}/licenses/${_platform}/LICENSE.txt"
    "fixture ${_platform} runtime terms\n")
endforeach()
file(WRITE "${_torch_notice_root}/licenses/common/LICENSE.txt"
  "fixture companion release notices\n")
set(_torch_sdk "${CMAKE_CURRENT_BINARY_DIR}/torch-sdk")
file(WRITE "${_torch_sdk}/share/doc/dnnl/LICENSE" "fixture bundled oneDNN terms\n")
file(WRITE "${_torch_sdk}/include/not-a-notice.h" "must not ship as a notice\n")
hyle_install_torch_sdk_notices("${_torch_notice_source}" "${_torch_sdk}")
include("${HYLE_SOURCE_DIR}/cmake/HylePackage.cmake")
hyle_install_package()
'''


class PackageComponentsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        scratch = ROOT / ".scratch/validation/package-components"
        scratch.mkdir(parents=True, exist_ok=True)
        cls.temp = tempfile.TemporaryDirectory(dir=scratch)
        cls.directory = Path(cls.temp.name)
        cls.source = cls.directory / "producer"
        cls.source.mkdir()
        (cls.source / "CMakeLists.txt").write_text(
            PRODUCER.replace("@ROOT@", ROOT.as_posix()), encoding="utf-8")
        cls.dependencies = cls.directory / "dependencies"
        for package in ("PackageCpuSDK", "PackageStaticSDK", "PackageTraceSDK",
                        "PackageHeadersSDK", "PackageSchedulerSDK"):
            destination = cls.dependencies / package
            destination.mkdir(parents=True)
            (destination / f"{package}Config.cmake").write_text(
                f"if(NOT TARGET {package}::runtime)\n"
                f"add_library({package}::runtime INTERFACE IMPORTED)\nendif()\n"
                f"set({package}_FOUND TRUE)\n", encoding="utf-8")
        cls.prefixes = {}
        for variant, static, trace in (("static", "ON", "OFF"),
                                       ("shared", "OFF", "OFF"),
                                       ("trace", "ON", "ON")):
            build = cls.directory / f"build-{variant}"
            prefix = cls.directory / f"sdk-{variant}"
            cls.command("cmake", "-S", cls.source, "-B", build,
                        "-DHYLE_STATIC=" + static, "-DSHIP_TRACE=" + trace,
                        "-DCMAKE_INSTALL_PREFIX=" + str(prefix))
            cls.command("cmake", "--install", build)
            cls.prefixes[variant] = prefix
        # Two additional tiny interface-only installations model equal-recipe
        # SDK copies and an SDK updated in place; no native libraries are copied.
        for variant in ("other-static", "mutable"):
            prefix = cls.directory / ("sdk-" + variant)
            cls.command("cmake", "--install", cls.directory / "build-static",
                        "--prefix", prefix)
            cls.prefixes[variant] = prefix

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def command(cls, *arguments, success=True):
        result = subprocess.run([str(item) for item in arguments],
                                capture_output=True, text=True, timeout=30,
                                **NO_WINDOW)
        if (result.returncode == 0) != success:
            raise AssertionError(result.stdout + result.stderr)
        return result.stdout + result.stderr

    def consume(self, name, body, *, variant="static", success=True):
        source = self.directory / ("consumer-" + name)
        source.mkdir()
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\n'
            'project(ConsumeHyle NONE)\n'
            'set(CMAKE_DISABLE_FIND_PACKAGE_PackageFixtureSDK TRUE)\n' + body,
            encoding="utf-8")
        return self.command(
            "cmake", "-S", source, "-B", source / "build",
            "-DHyle_DIR=" + str(self.prefixes[variant] / "lib/cmake/Hyle"),
            "-DCMAKE_PREFIX_PATH=" + str(self.dependencies), success=success)

    def test_project_and_dependency_notices_ship_in_each_component(self):
        notices = {"HYLE-LICENSE": (ROOT / "LICENSE").read_bytes(),
                   "HYLE-NOTICE": (ROOT / "NOTICE").read_bytes(),
                   "THIRD_PARTY-NOTICES.md":
                       (ROOT / "THIRD_PARTY_NOTICES.md").read_bytes(),
                   "fixture-LICENSE":
                       (self.directory / "build-static/fixture-LICENSE").read_bytes()}
        prefixes = list(self.prefixes.values())
        for variant in ("static", "shared"):
            for component in ("Hyle_Runtime", "Hyle_Development"):
                prefix = self.directory / f"notices-{variant}-{component}"
                self.command("cmake", "--install", self.directory / f"build-{variant}",
                             "--prefix", prefix, "--component", component)
                prefixes.append(prefix)
        for prefix in prefixes:
            for name, expected in notices.items():
                with self.subTest(prefix=prefix.name, notice=name):
                    self.assertEqual(
                        (prefix / "share/licenses/Hyle" / name).read_bytes(), expected)
            platform = "windows" if os.name == "nt" else "macos" if sys.platform == "darwin" else "linux"
            source = self.directory / "build-static/torch-notice-source/third_party/torch_sdk"
            installed = prefix / "share/licenses/Hyle/torch-sdk"
            for name in ("LICENSE", "NOTICE", "ORIGIN.json", f"licenses/{platform}/LICENSE.txt",
                         "licenses/common/LICENSE.txt"):
                with self.subTest(prefix=prefix.name, torch_notice=name):
                    self.assertEqual((installed / name).read_bytes(), (source / name).read_bytes())
            for other in {"windows", "linux", "macos"} - {platform}:
                self.assertFalse((installed / "licenses" / other).exists())
            original = self.directory / "build-static/torch-sdk/share/doc/dnnl/LICENSE"
            self.assertEqual(
                (prefix / "share/licenses/Hyle/torch-cpu-sdk/share/doc/dnnl/LICENSE").read_bytes(),
                original.read_bytes())
            self.assertFalse((prefix / "share/licenses/Hyle/torch-cpu-sdk/include").exists())

    def test_mem_import_does_not_probe_or_import_compute(self):
        for variant in ("static", "shared"):
            self.consume("mem-" + variant, r'''
set(CMAKE_DISABLE_FIND_PACKAGE_PackageCpuSDK TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_PackageStaticSDK TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_PackageTraceSDK TRUE)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
if(NOT TARGET hyle::mem OR NOT TARGET hyle::core OR TARGET hyle::compute
   OR TARGET hyle::_factor_compute_backend_cpu OR TARGET PackageCpuSDK::runtime)
  message(FATAL_ERROR "mem import acquired the wrong target closure")
endif()
if(NOT Hyle_LOADED_COMPONENTS STREQUAL "core;mem"
   OR NOT "compute" IN_LIST Hyle_AVAILABLE_COMPONENTS
   OR NOT "compute_cpu" IN_LIST Hyle_AVAILABLE_FEATURES
   OR NOT "compute_ggml_cpu" IN_LIST Hyle_AVAILABLE_FEATURES)
  message(FATAL_ERROR "available/loaded metadata is inconsistent")
endif()
''', variant=variant)

    def test_hyle_dir_alone_finds_bundled_private_shims(self):
        self.consume("private-shim-dir", r'''
set(CMAKE_PREFIX_PATH "")
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core)
if(NOT TARGET HyleFixtureRuntime::runtime OR NOT TARGET hyle::core)
  message(FATAL_ERROR "Hyle_DIR did not locate this SDK's private shim")
endif()
''')

    def test_graph_progress_service_does_not_pollute_graph_import(self):
        for variant in ("static", "shared"):
            self.consume("graph-only-" + variant, r'''
set(CMAKE_DISABLE_FIND_PACKAGE_PackageSchedulerSDK TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_PackageCpuSDK TRUE)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS graph)
if(NOT TARGET hyle::graph OR TARGET hyle::sched OR TARGET hyle::graph_progress
   OR TARGET hyle::_factor_sched OR TARGET hyle::_factor_graph_progress
   OR TARGET PackageSchedulerSDK::runtime)
  message(FATAL_ERROR "Graph imported an unrequested progress service")
endif()
''', variant=variant)
            self.consume("graph-progress-" + variant, r'''
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS graph sched)
if(TARGET hyle::graph_progress OR TARGET hyle::_factor_graph_progress)
  message(FATAL_ERROR "Graph + Sched implicitly imported progress")
endif()
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS graph_progress)
if(NOT TARGET hyle::graph_progress OR NOT TARGET hyle::_factor_graph_progress
   OR NOT TARGET hyle::sched OR NOT TARGET PackageSchedulerSDK::runtime)
  message(FATAL_ERROR "The explicit service omitted its public Scheduler dependency")
endif()
''', variant=variant)

    def test_failed_quiet_compute_probe_allows_mem_retry_from_same_sdk(self):
        self.consume("quiet-retry", r'''
set(CMAKE_DISABLE_FIND_PACKAGE_PackageCpuSDK TRUE)
find_package(Hyle 0.1.0 EXACT QUIET COMPONENTS compute)
if(Hyle_FOUND)
  message(FATAL_ERROR "missing external SDK did not reject Compute")
endif()
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
if(NOT TARGET hyle::mem OR NOT Hyle_mem_FOUND)
  message(FATAL_ERROR "the failed Compute probe prevented Mem fallback")
endif()
''')

    def test_missing_private_shim_propagates_failure_and_allows_retry(self):
        self.consume("private-shim-retry", r'''
set(CMAKE_DISABLE_FIND_PACKAGE_HyleFixtureRuntime TRUE)
find_package(Hyle 0.1.0 EXACT QUIET COMPONENTS core)
if(Hyle_FOUND)
  message(FATAL_ERROR "a missing private shim was reported as successful")
endif()
unset(CMAKE_DISABLE_FIND_PACKAGE_HyleFixtureRuntime)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core)
if(NOT TARGET hyle::core)
  message(FATAL_ERROR "Core could not be retried after restoring its shim")
endif()
''')

    def test_second_find_adds_compute_and_preserves_public_properties(self):
        for variant in ("static", "shared"):
            self.consume("repeat-" + variant, r'''
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute)
get_target_property(routes hyle::compute HYLE_COMPUTE_ROUTES)
if(NOT routes STREQUAL "cpu" OR NOT TARGET hyle::_fake_cpu_vendor
   OR NOT TARGET PackageCpuSDK::runtime
   OR NOT Hyle_LOADED_COMPONENTS STREQUAL "core;mem;compute")
  message(FATAL_ERROR "incremental import lost its closure or properties")
endif()
if(Hyle_STATIC AND NOT TARGET PackageStaticSDK::runtime)
  message(FATAL_ERROR "static-only dependency was omitted")
elseif(NOT Hyle_STATIC AND TARGET PackageStaticSDK::runtime)
  message(FATAL_ERROR "shared import acquired a static-only dependency")
endif()
''', variant=variant)

    def test_unknown_required_component_is_rejected_before_dependency_probes(self):
        output = self.consume("missing", r'''
set(CMAKE_DISABLE_FIND_PACKAGE_PackageCpuSDK TRUE)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS missing)
''', success=False)
        self.assertIn("missing requested component(s): missing", output)
        self.assertNotIn("find_package for module PackageCpuSDK", output)

    def test_default_import_loads_all_products(self):
        self.consume("all", r'''
find_package(Hyle 0.1.0 EXACT REQUIRED)
if(NOT Hyle_LOADED_COMPONENTS STREQUAL Hyle_AVAILABLE_COMPONENTS)
  message(FATAL_ERROR "default import did not load all public products")
endif()
''')

    def test_missing_optional_component_keeps_valid_requested_products(self):
        self.consume("optional", r'''
set(CMAKE_DISABLE_FIND_PACKAGE_PackageCpuSDK TRUE)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem OPTIONAL_COMPONENTS missing)
if(NOT TARGET hyle::mem OR NOT Hyle_mem_FOUND OR Hyle_missing_FOUND)
  message(FATAL_ERROR "optional component prevented a valid product import")
endif()
''')

    def test_shared_header_dependency_follows_its_component_users(self):
        self.consume("headers-mem", r'''
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
if(NOT PackageHeadersSDK_FOUND)
  message(FATAL_ERROR "shared header dependency was omitted")
endif()
''')

    def test_source_and_installed_targets_cannot_be_mixed(self):
        output = self.consume("source-origin", r'''
add_library(source_contract INTERFACE)
add_library(hyle::_build_contract ALIAS source_contract)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
''', success=False)
        self.assertIn("embedded source build", " ".join(output.split()))

    def test_two_sdk_prefixes_with_the_same_recipe_cannot_be_mixed(self):
        other = self.prefixes["other-static"] / "lib/cmake/Hyle"
        output = self.consume("sdk-origin", r'''
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
set(Hyle_DIR "@OTHER@")
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute)
'''.replace("@OTHER@", other.as_posix()), success=False)
        self.assertIn("another SDK configuration or recipe", " ".join(output.split()))

    def test_changed_recipe_at_the_same_prefix_is_rejected(self):
        output = self.consume("recipe-origin", r'''
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
get_target_property(old_recipe hyle::_build_contract HYLE_PACKAGE_RECIPE_ID)
file(READ "${Hyle_DIR}/HyleConfig.cmake" config)
string(REPLACE "${old_recipe}" "changed-fixture-recipe" config "${config}")
file(WRITE "${Hyle_DIR}/HyleConfig.cmake" "${config}")
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute)
''', variant="mutable", success=False)
        self.assertIn("another SDK configuration or recipe", " ".join(output.split()))

    def test_sibling_directories_can_use_different_sdk_origins(self):
        first = self.prefixes["static"] / "lib/cmake/Hyle"
        second = self.prefixes["shared"] / "lib/cmake/Hyle"
        self.consume("sibling-origins", r'''
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/left" "${CMAKE_CURRENT_BINARY_DIR}/right")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/left/CMakeLists.txt" [=[
set(Hyle_DIR "@FIRST@")
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
get_target_property(origin HyleFixtureRuntime::runtime HYLE_FIXTURE_ORIGIN)
get_filename_component(expected "@FIRST@/../HyleFixtureRuntime" ABSOLUTE)
if(NOT origin STREQUAL expected)
  message(FATAL_ERROR "left sibling loaded another SDK's private shim: ${origin}")
endif()
]=])
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/right/CMakeLists.txt" [=[
set(Hyle_DIR "@SECOND@")
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute)
get_target_property(origin HyleFixtureRuntime::runtime HYLE_FIXTURE_ORIGIN)
get_filename_component(expected "@SECOND@/../HyleFixtureRuntime" ABSOLUTE)
if(NOT origin STREQUAL expected)
  message(FATAL_ERROR "right sibling loaded another SDK's private shim: ${origin}")
endif()
]=])
add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/left" left-build)
add_subdirectory("${CMAKE_CURRENT_BINARY_DIR}/right" right-build)
'''.replace("@FIRST@", first.as_posix()).replace("@SECOND@", second.as_posix()))
        self.consume("headers-compute", r'''
set(CMAKE_DISABLE_FIND_PACKAGE_PackageHeadersSDK TRUE)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS compute)
if(PackageHeadersSDK_FOUND)
  message(FATAL_ERROR "shared header dependency leaked into another product")
endif()
''')

    def test_test_only_targets_headers_and_assets_do_not_ship(self):
        for variant, prefix in self.prefixes.items():
            with self.subTest(variant=variant):
                config = (prefix / "lib/cmake/Hyle/HyleConfig.cmake").read_text()
                index = (prefix / "lib/cmake/Hyle/HylePackageIndex.cmake").read_text()
                self.assertNotIn("fixture_only", config + index)
                self.assertFalse((prefix / "include/hyle/util/function_ref.h").exists())
                self.assertFalse((prefix / "share/fixture-only").exists())
                self.assertFalse((prefix / "share/fixture-headers").exists())
                self.assertTrue((prefix / "share/shared-headers/shared-header.txt").exists())
                if variant != "trace":
                    self.assertNotIn("PackageTraceSDK", index)
                    self.assertFalse((prefix / "include/hyle/obs/trace.h").exists())

    def test_reachable_foundation_dependency_is_retained(self):
        self.consume("trace", r'''
set(CMAKE_DISABLE_FIND_PACKAGE_PackageCpuSDK TRUE)
find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS mem)
if(NOT TARGET PackageTraceSDK::runtime OR NOT TARGET hyle::_fake_trace_vendor
   OR NOT "trace" IN_LIST Hyle_AVAILABLE_FEATURES)
  message(FATAL_ERROR "reachable foundation dependency was omitted")
endif()
''', variant="trace")


if __name__ == "__main__":
    unittest.main()
