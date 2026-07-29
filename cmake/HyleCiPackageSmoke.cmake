# Cross-platform binary-package smoke used by CI and maintainers. It creates
# shared and static all-component CPU SDKs, then configures/builds/runs
# source-independent consumers against each install prefix. The focused mem C
# and C++ consumers exercise a real API and loader/static-link closure; the
# per-component consumers mechanically prove that every exported target can be
# discovered and linked without creating an unnatural duplicate closure.
# Optional SDK/hardware feature matrices remain release checks, not per-commit
# fan heaters.

cmake_minimum_required(VERSION 3.25)

if(NOT DEFINED HYLE_SOURCE_DIR OR HYLE_SOURCE_DIR STREQUAL "")
  message(FATAL_ERROR "Pass -DHYLE_SOURCE_DIR=<HYLE checkout>")
endif()
if(NOT DEFINED HYLE_SMOKE_ROOT OR HYLE_SMOKE_ROOT STREQUAL "")
  message(FATAL_ERROR "Pass -DHYLE_SMOKE_ROOT=<scratch directory>")
endif()

get_filename_component(HYLE_SOURCE_DIR "${HYLE_SOURCE_DIR}" ABSOLUTE)
get_filename_component(HYLE_SMOKE_ROOT "${HYLE_SMOKE_ROOT}" ABSOLUTE)

file(STRINGS "${HYLE_SOURCE_DIR}/include/hyle/core/version_defs.h"
  _hyle_version_lines REGEX
  "^#define HYLE_VERSION_(MAJOR|MINOR|PATCH) [0-9]+$")
foreach(_part IN ITEMS MAJOR MINOR PATCH)
  foreach(_line IN LISTS _hyle_version_lines)
    if(_line MATCHES "^#define HYLE_VERSION_${_part} ([0-9]+)$")
      set(_hyle_version_${_part} "${CMAKE_MATCH_1}")
    endif()
  endforeach()
  if(NOT DEFINED _hyle_version_${_part})
    message(FATAL_ERROR "Cannot read HYLE_VERSION_${_part} from version_defs.h")
  endif()
endforeach()
set(_hyle_compatible_version
  "${_hyle_version_MAJOR}.${_hyle_version_MINOR}")
set(_hyle_exact_version
  "${_hyle_version_MAJOR}.${_hyle_version_MINOR}.${_hyle_version_PATCH}")
if(_hyle_version_MAJOR EQUAL 0)
  math(EXPR _hyle_next_minor "${_hyle_version_MINOR} + 1")
  set(_hyle_incompatible_version
    "${_hyle_version_MAJOR}.${_hyle_next_minor}")
else()
  math(EXPR _hyle_next_major "${_hyle_version_MAJOR} + 1")
  set(_hyle_incompatible_version "${_hyle_next_major}.0")
endif()

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
  set(_components all)
  set(_build "${HYLE_SMOKE_ROOT}/package-${_linkage}")
  set(_prefix "${HYLE_SMOKE_ROOT}/prefix-${_linkage}")
  set(_consumer "${HYLE_SMOKE_ROOT}/consumer-${_linkage}")
  file(REMOVE_RECURSE "${_build}" "${_prefix}" "${_consumer}")

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
    -DHYLE_IO_HTTP2=OFF
    -DHYLE_COMPUTE_CUDA=OFF
    -DHYLE_COMPUTE_GGML=OFF
    -DHYLE_COMPUTE_VULKAN=OFF)
  _hyle_smoke_run("build ${_linkage} SDK"
    "${CMAKE_COMMAND}" --build "${_build}" --config Release --parallel 2)
  _hyle_smoke_run("install ${_linkage} SDK"
    "${CMAKE_COMMAND}" --install "${_build}" --config Release)
  file(GLOB _hyle_installed_version_files
    "${_prefix}/*/cmake/Hyle/HyleConfigVersion.cmake")
  list(LENGTH _hyle_installed_version_files _hyle_version_file_count)
  if(NOT _hyle_version_file_count EQUAL 1)
    message(FATAL_ERROR
      "expected one installed HyleConfigVersion.cmake under ${_prefix}, found "
      "${_hyle_version_file_count}")
  endif()
  list(GET _hyle_installed_version_files 0 _hyle_installed_version_file)
  _hyle_smoke_run("check ${_linkage} package version contract"
    "${CMAKE_COMMAND}"
    "-DHYLE_VERSION_FILE:FILEPATH=${_hyle_installed_version_file}"
    "-DHYLE_EXACT_VERSION:STRING=${_hyle_exact_version}"
    "-DHYLE_COMPATIBLE_VERSION:STRING=${_hyle_compatible_version}"
    "-DHYLE_INCOMPATIBLE_VERSION:STRING=${_hyle_incompatible_version}"
    -P "${HYLE_SOURCE_DIR}/cmake/HyleCheckPackageVersion.cmake")

  _hyle_smoke_run("configure ${_linkage} installed consumers"
    "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}/samples/05_installed_c"
    -B "${_consumer}"
    ${_hyle_generator_args}
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH=${_prefix})
  _hyle_smoke_run("build ${_linkage} installed consumers"
    "${CMAKE_COMMAND}" --build "${_consumer}" --config Release --parallel 2)
  _hyle_smoke_run("run ${_linkage} installed consumers"
    "${CMAKE_CTEST_COMMAND}" --test-dir "${_consumer}" -C Release
    --output-on-failure)

endforeach()
