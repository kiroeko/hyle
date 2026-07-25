# Cross-platform binary-package smoke used by CI and maintainers. It creates
# shared and static mem SDKs, then configures/builds/runs one source-independent
# pure-C error+mem consumer against each install prefix.  Shared mem exercises
# the mem -> platform -> error loader chain; static mem exercises the private
# mimalloc archive and C++ link-driver closure. Keep this bounded: full
# simd/compute/io delivery matrices are release checks, not per-commit fan
# heaters.

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED HYLE_SOURCE_DIR OR HYLE_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "Pass -DHYLE_SOURCE_DIR=<HYLE checkout>")
endif()
if(NOT DEFINED HYLE_SMOKE_ROOT OR HYLE_SMOKE_ROOT STREQUAL "")
  message(FATAL_ERROR "Pass -DHYLE_SMOKE_ROOT=<scratch directory>")
endif()

get_filename_component(HYLE_SOURCE_DIR "${HYLE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(HYLE_SMOKE_ROOT "${HYLE_SMOKE_ROOT}" ABSOLUTE)

function(_hyle_smoke_run label)
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

foreach(_linkage IN ITEMS shared static)
  if(_linkage STREQUAL "static")
    set(_static ON)
  else()
    set(_static OFF)
  endif()
  set(_components mem)
  set(_build "${HYLE_SMOKE_ROOT}/package-${_linkage}")
  set(_prefix "${HYLE_SMOKE_ROOT}/prefix-${_linkage}")
  set(_consumer "${HYLE_SMOKE_ROOT}/consumer-${_linkage}")

  _hyle_smoke_run("configure ${_linkage} SDK"
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}" -B "${_build}"
    ${_hyle_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${_prefix}
    -DHYLE_COMPONENTS=${_components}
    -DHYLE_BUILD_TESTS=OFF
    -DHYLE_BUILD_BENCH=OFF
    -DHYLE_INSTALL=ON
    -DHYLE_STATIC=${_static}
    -DHYLE_IO_TLS=OFF
    -DHYLE_COMPUTE_CUDA=OFF)
  _hyle_smoke_run("build ${_linkage} SDK"
    "${CMAKE_COMMAND}" --build "${_build}" --config Release --parallel 2)
  _hyle_smoke_run("install ${_linkage} SDK"
    "${CMAKE_COMMAND}" --install "${_build}" --config Release)

  _hyle_smoke_run("configure ${_linkage} pure-C consumer"
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}/samples/05_installed_c"
    -B "${_consumer}"
    ${_hyle_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=${_prefix})
  _hyle_smoke_run("build ${_linkage} pure-C consumer"
    "${CMAKE_COMMAND}" --build "${_consumer}" --config Release --parallel 2)
  _hyle_smoke_run("run ${_linkage} pure-C consumer"
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer}" -C Release
    --output-on-failure)

endforeach()
