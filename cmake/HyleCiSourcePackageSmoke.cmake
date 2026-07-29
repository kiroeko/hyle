# End-to-end verification for HYLE's complete-source delivery shape: generate
# from a clean Git commit, validate digest/manifest/legal payload, extract, then
# configure and build a component from the archive without the checkout.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_PACKAGE_BUILD HYLE_SMOKE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=<path>")
  endif()
  get_filename_component(${_required} "${${_required}}" ABSOLUTE)
endforeach()

function(_hyle_source_run label)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    COMMAND_ECHO STDOUT)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "${label} failed with exit code ${_result}")
  endif()
endfunction()

_hyle_source_run("generate complete-source package"
  "${CMAKE_COMMAND}" --build "${HYLE_PACKAGE_BUILD}" --config Release
  --target hyle_source_package --parallel 2)

file(GLOB _archives
  "${HYLE_PACKAGE_BUILD}/dist/hyle-*-source.zip")
list(LENGTH _archives _archive_count)
if(NOT _archive_count EQUAL 1)
  message(FATAL_ERROR
    "Expected exactly one source archive under ${HYLE_PACKAGE_BUILD}/dist, "
    "found ${_archive_count}: ${_archives}")
endif()
list(GET _archives 0 _archive)

if(NOT EXISTS "${_archive}.sha256")
  message(FATAL_ERROR "Missing source archive digest: ${_archive}.sha256")
endif()
file(SHA256 "${_archive}" _actual_sha256)
file(READ "${_archive}.sha256" _digest_record)
if(NOT _digest_record MATCHES "^${_actual_sha256}  hyle-.+-source\\.zip")
  message(FATAL_ERROR
    "Source archive SHA-256 record does not match ${_actual_sha256}: "
    "${_digest_record}")
endif()

find_package(Git REQUIRED)
execute_process(
  COMMAND "${GIT_EXECUTABLE}" -C "${HYLE_SOURCE_DIR}"
          rev-parse --verify HEAD
  RESULT_VARIABLE _git_result
  OUTPUT_VARIABLE _commit
  ERROR_VARIABLE _git_error
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _git_result EQUAL 0)
  message(FATAL_ERROR "Cannot resolve source checkout commit: ${_git_error}")
endif()

set(_extract_dir "${HYLE_SMOKE_ROOT}/source-package-extract")
file(MAKE_DIRECTORY "${_extract_dir}")
file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_extract_dir}")
file(GLOB _source_roots LIST_DIRECTORIES TRUE
  "${_extract_dir}/hyle-*-source")
list(LENGTH _source_roots _source_root_count)
if(NOT _source_root_count EQUAL 1)
  message(FATAL_ERROR
    "Expected one extracted source root, found ${_source_root_count}: "
    "${_source_roots}")
endif()
list(GET _source_roots 0 _source_root)

foreach(_payload IN ITEMS CMakeLists.txt SOURCE-MANIFEST.txt LICENSE
                          THIRD_PARTY_NOTICES.md)
  if(NOT EXISTS "${_source_root}/${_payload}")
    message(FATAL_ERROR "Source package is missing ${_payload}")
  endif()
endforeach()
file(READ "${_source_root}/SOURCE-MANIFEST.txt" _manifest)
if(NOT _manifest MATCHES "Git commit: ${_commit}")
  message(FATAL_ERROR
    "Source manifest does not name checkout commit ${_commit}:\n${_manifest}")
endif()

set(_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()
set(_consumer_build "${HYLE_SMOKE_ROOT}/source-package-build")
_hyle_source_run("configure extracted source package"
  "${CMAKE_COMMAND}" -S "${_source_root}" -B "${_consumer_build}"
  ${_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DHYLE_COMPONENTS=core
  -DHYLE_BUILD_TESTS=OFF
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_WERROR=ON
  -DHYLE_IO_TLS=OFF
  -DHYLE_COMPUTE_CUDA=OFF)
_hyle_source_run("build extracted source package"
  "${CMAKE_COMMAND}" --build "${_consumer_build}" --config Release
  --target hyle_core --parallel 2)

message(STATUS
  "HYLE complete-source package passed: ${_archive} (${_commit})")
