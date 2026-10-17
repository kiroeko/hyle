"""Public product links and test-only macros; configure fixtures only."""

from pathlib import Path
import os
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / ".scratch/build-system-followup/product-macro-contracts"
NO_WINDOW = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}


class ProductMacroContractTest(unittest.TestCase):
    def setUp(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.directory = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def configure(self, name, source, *, language="NONE"):
        directory = self.directory / name
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.25)\n'
            f'project(ProductMacroFixture LANGUAGES {language})\n'
            f'set(HYLE_SOURCE_DIR "{ROOT.as_posix()}")\n' + source,
            encoding="utf-8")
        result = subprocess.run(
            ["cmake", "-S", str(directory), "-B", str(directory / "build")],
            capture_output=True, text=True, timeout=20, **NO_WINDOW)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return directory / "build"

    def test_actual_cpu_ggml_presence_reaches_private_route_dependencies(self):
        build = self.configure("cpu-ggml-presence", r'''
include("${HYLE_SOURCE_DIR}/cmake/HyleComputeCapabilities.cmake")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/probe.cc" "int probe() { return 0; }\n")
add_library(ggml INTERFACE)
set(HYLE_GGML_CPU_ENABLED ON)
# Load the owning declaration without configuring or compiling its SDK/math.
function(hyle_add_module name)
  add_library(hyle_${name} STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/probe.cc")
endfunction()
include("${HYLE_SOURCE_DIR}/src/compute_provider_ggml/CMakeLists.txt")
get_target_property(capabilities hyle_compute_provider_ggml HYLE_COMPUTE_CAPABILITIES)
if(NOT ggml_cpu IN_LIST capabilities)
  message(FATAL_ERROR "Actual ggml CPU provider lost its capability")
endif()
# Both real route contexts use this presence macro and privately link this
# provider. Verify compile usage, not just package feature metadata.
foreach(route IN ITEMS compute_route_cpu graph_route_cpu)
  add_library(hyle_${route} STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_BINARY_DIR}/probe.cc")
  target_link_libraries(hyle_${route} PRIVATE hyle_compute_provider_ggml)
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${route}-defines.txt"
    CONTENT "$<TARGET_PROPERTY:hyle_${route},COMPILE_DEFINITIONS>\n")
endforeach()
''', language="CXX")
        for route in ("compute_route_cpu", "graph_route_cpu"):
            definitions = (build / (route + "-defines.txt")).read_text()
            self.assertIn("HYLE_COMPUTE_GGML_CPU=1", definitions)

    def test_cuda_experts_are_public_product_roots(self):
        for system in ("Windows", "Linux"):
            with self.subTest(system=system):
                build = self.configure("cuda-experts-" + system, r'''
set(CMAKE_SYSTEM_NAME "@SYSTEM@")
set(HYLE_COMPONENTS "compute;graph")
set(HYLE_COMPUTE_ROUTES cuda)
set(HYLE_BUILD_TESTS OFF)
set(HYLE_TARGET_PROCESSOR x86_64)
set(HYLE_FCONTEXT_SUPPORTED TRUE)
include("${HYLE_SOURCE_DIR}/cmake/HyleComponents.cmake")
include("${HYLE_SOURCE_DIR}/cmake/HyleModule.cmake")
foreach(factor IN LISTS HYLE_RESOLVED_COMPONENT_LIST)
  add_library(hyle_${factor} INTERFACE)
endforeach()
foreach(factor IN LISTS HYLE_RESOLVED_COMPONENT_LIST)
  string(TOUPPER "${factor}" upper)
  foreach(dep IN LISTS _HYLE_PUBLIC_DEPS_${upper})
    target_link_libraries(hyle_${factor} INTERFACE hyle_${dep})
  endforeach()
  foreach(dep IN LISTS _HYLE_PRIVATE_DEPS_${upper})
    target_link_libraries(hyle_${factor} INTERFACE "$<LINK_ONLY:hyle_${dep}>")
  endforeach()
endforeach()
foreach(factor IN ITEMS compute_backend_cuda compute_provider_cublas compute_provider_cuda)
  hyle_target_compute_capability(hyle_${factor} cuda VISIBILITY INTERFACE)
endforeach()
foreach(factor IN ITEMS compute_provider_ggml_cuda graph_provider_ggml_cuda)
  hyle_target_compute_capability(hyle_${factor} ggml_cuda VISIBILITY INTERFACE)
endforeach()
hyle_target_compute_capability(hyle_compute_provider_torch_cuda torch_cuda VISIBILITY INTERFACE)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  hyle_target_compute_capability(hyle_compute_provider_flashinfer flashinfer VISIBILITY INTERFACE)
else()
  set_property(TARGET hyle_compute_provider_flashinfer PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
hyle_add_product_facades()
foreach(product IN ITEMS compute graph)
  get_target_property(links hyle_product_${product} INTERFACE_LINK_LIBRARIES)
  foreach(expert IN ITEMS compute_provider_cublas compute_provider_cuda
      compute_provider_ggml_cuda compute_provider_torch_cuda)
    if(NOT "hyle_${expert}" IN_LIST links)
      message(FATAL_ERROR "${product} does not publicly link ${expert}: ${links}")
    endif()
  endforeach()
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(NOT hyle_compute_provider_flashinfer IN_LIST links)
      message(FATAL_ERROR "Linux ${product} lost its FlashInfer expert")
    endif()
  elseif(hyle_compute_provider_flashinfer IN_LIST links)
    message(FATAL_ERROR "${product} advertises absent FlashInfer")
  endif()
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${product}-defines.txt"
    CONTENT "$<TARGET_PROPERTY:hyle_product_${product},INTERFACE_COMPILE_DEFINITIONS>\n")
endforeach()
get_target_property(graph_links hyle_product_graph INTERFACE_LINK_LIBRARIES)
if(NOT hyle_graph_provider_ggml_cuda IN_LIST graph_links)
  message(FATAL_ERROR "Graph lost its public region-provider factory")
endif()
foreach(expert IN ITEMS compute_provider_cublas compute_provider_cuda
    compute_provider_ggml_cuda compute_provider_torch_cuda)
  if(NOT expert IN_LIST _HYLE_PRIVATE_DEPS_COMPUTE_ROUTE_CUDA OR
      expert IN_LIST _HYLE_PUBLIC_DEPS_COMPUTE_ROUTE_CUDA)
    message(FATAL_ERROR "Route implementation edge changed for ${expert}")
  endif()
endforeach()
'''.replace("@SYSTEM@", system))
                for product in ("compute", "graph"):
                    definitions = (build / (product + "-defines.txt")).read_text()
                    for capability in ("CUDA", "GGML_CUDA", "TORCH_CUDA"):
                        self.assertIn("HYLE_COMPUTE_" + capability + "=1", definitions)
                    self.assertEqual("HYLE_COMPUTE_FLASHINFER=1" in definitions,
                                     system == "Linux")

    def test_ordinary_simd_tests_do_not_add_private_exports(self):
        for instrumentation in ("OFF", "ON"):
            with self.subTest(instrumentation=instrumentation):
                self.configure("simd-instrumentation-" + instrumentation, r'''
set(HYLE_TEST_INSTRUMENTATION @INSTRUMENTATION@)
set(HYLE_TARGET_PROCESSOR generic)
add_library(hyle_simd INTERFACE)
# Record private target metadata without compiling any native SIMD or GTest.
function(hyle_add_test module)
  cmake_parse_arguments(ARG "NO_AUTORUN" "NAME" "SOURCES;DEFINES;DEPS;OPTIONS" ${ARGN})
  if(ARG_NAME)
    set(target hyle_${module}_${ARG_NAME}_test)
  else()
    set(target hyle_${module}_test)
  endif()
  add_library(${target} INTERFACE)
  set_property(TARGET ${target} PROPERTY PROBE_SOURCES "${ARG_SOURCES}")
endfunction()
function(target_compile_definitions target visibility)
  set_property(TARGET ${target} APPEND PROPERTY PROBE_DEFINITIONS ${ARGN})
endfunction()
function(target_include_directories)
endfunction()
function(hyle_add_fat_kernel)
endfunction()
include("${HYLE_SOURCE_DIR}/tests/simd/CMakeLists.txt")
foreach(target IN ITEMS hyle_simd hyle_simd_test)
  get_target_property(definitions ${target} PROBE_DEFINITIONS)
  if(HYLE_TEST_INSTRUMENTATION)
    if(NOT "HYLE_SIMD_INTERNAL_TESTING=1" IN_LIST definitions)
      message(FATAL_ERROR "Instrumented ${target} lost its internal access")
    endif()
  elseif("HYLE_SIMD_INTERNAL_TESTING=1" IN_LIST definitions)
    message(FATAL_ERROR "Ordinary tests change ${target} exports")
  endif()
endforeach()
get_target_property(sources hyle_simd_test PROBE_SOURCES)
foreach(source IN ITEMS runtime_test.cc ops_test.cc search_test.cc scan_test.cc
    dtype_test.cc math_test.cc sort_test.cc bitpack_test.cc narrow_ops_test.cc)
  if(NOT source IN_LIST sources)
    message(FATAL_ERROR "Public behavior tests disappeared: ${source}")
  endif()
endforeach()
'''.replace("@INSTRUMENTATION@", instrumentation))


if __name__ == "__main__":
    unittest.main()
