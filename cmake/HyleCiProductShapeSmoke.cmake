# Cross-platform shipping-layout test smoke used by CI and maintainers.  The
# ordinary suite runs with HYLE_TESTING undefined; instrumented fault/structure
# tests remain a separate, additive configuration.

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED HYLE_SOURCE_DIR OR HYLE_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "Pass -DHYLE_SOURCE_DIR=<HYLE checkout>")
endif()
if(NOT DEFINED HYLE_SMOKE_ROOT OR HYLE_SMOKE_ROOT STREQUAL "")
  message(FATAL_ERROR "Pass -DHYLE_SMOKE_ROOT=<scratch directory>")
endif()

get_filename_component(HYLE_SOURCE_DIR "${HYLE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(HYLE_SMOKE_ROOT "${HYLE_SMOKE_ROOT}" ABSOLUTE)

function(_hyle_product_shape_run label)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    COMMAND_ECHO STDOUT)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${label} failed with exit code ${_result}")
  endif()
endfunction()

set(_hyle_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _hyle_generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()

set(_hyle_dependency_args "")
if(DEFINED HYLE_GOOGLETEST_SOURCE AND NOT HYLE_GOOGLETEST_SOURCE STREQUAL "")
  get_filename_component(HYLE_GOOGLETEST_SOURCE
    "${HYLE_GOOGLETEST_SOURCE}" ABSOLUTE)
  if(NOT EXISTS "${HYLE_GOOGLETEST_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR
      "HYLE_GOOGLETEST_SOURCE is not a GoogleTest source tree: "
      "${HYLE_GOOGLETEST_SOURCE}")
  endif()
  list(APPEND _hyle_dependency_args
    "-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${HYLE_GOOGLETEST_SOURCE}")
endif()

set(_hyle_build "${HYLE_SMOKE_ROOT}/product-shape")
file(REMOVE_RECURSE "${_hyle_build}")
# Only roots whose production-layout contract has already closed belong here.
# Their dependency closure necessarily pulls in some evolving modules (notably
# container through util). Passing this smoke proves those transitive seams are
# healthy; it does not certify the evolving module's complete product surface.
# Keep the strictly closed modules as product roots rather than spelling the
# transitive graph by hand, so the smoke also proves that the central component
# manifest resolves the real shipping closure correctly.
set(_hyle_shipping_roots core util obs time)
list(JOIN _hyle_shipping_roots "," _hyle_shipping_components)
_hyle_product_shape_run("configure production-shape tests"
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}" -B "${_hyle_build}"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DHYLE_COMPONENTS=${_hyle_shipping_components}
  -DHYLE_BUILD_TESTS=ON
  -DHYLE_TEST_INSTRUMENTATION=OFF
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_WERROR=ON
  -DHYLE_IO_TLS=OFF
  -DHYLE_COMPUTE_CUDA=OFF
  ${_hyle_dependency_args})
_hyle_product_shape_run("build production-shape tests"
  "${CMAKE_COMMAND}" --build "${_hyle_build}" --config Release
  --target hyle_tests
  --parallel 2)
_hyle_product_shape_run("run production-shape tests"
  "${CMAKE_CTEST_COMMAND}" --test-dir "${_hyle_build}" -C Release
  --output-on-failure)
