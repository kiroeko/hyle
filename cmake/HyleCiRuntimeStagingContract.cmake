cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT)
  if(NOT IS_ABSOLUTE "${${_required}}")
    message(FATAL_ERROR "${_required} must be an absolute path")
  endif()
  file(TO_CMAKE_PATH "${${_required}}" ${_required})
endforeach()
if(NOT WIN32)
  message(STATUS "Runtime staging contract is Windows-only")
  return()
endif()
if(HYLE_STAGING_PREFIX)
  file(TO_CMAKE_PATH "${HYLE_STAGING_PREFIX}" HYLE_STAGING_PREFIX)
endif()

function(_run)
  execute_process(COMMAND ${ARGN} RESULT_VARIABLE _result
    OUTPUT_VARIABLE _out ERROR_VARIABLE _err)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Runtime staging command failed: ${ARGN}\n${_out}\n${_err}")
  endif()
endfunction()

set(_source "${HYLE_SMOKE_ROOT}/source")
set(_build "${HYLE_SMOKE_ROOT}/build")
file(MAKE_DIRECTORY "${_source}")
set(_project [=[
cmake_minimum_required(VERSION 3.25)
project(runtime_staging_probe LANGUAGES CXX)
if("@HYLE_STAGING_PREFIX@" STREQUAL "")
  include("@HYLE_SOURCE_DIR@/cmake/HyleModule.cmake")
else()
  find_package(Hyle 0.1.0 EXACT REQUIRED COMPONENTS core
    PATHS "@HYLE_STAGING_PREFIX@" NO_DEFAULT_PATH)
endif()
add_library(staging_probe_dll SHARED library.cc)
target_compile_definitions(staging_probe_dll PRIVATE PROBE_VALUE=${PROBE_VALUE})
add_executable(runtime_probe main.cc)
target_link_libraries(runtime_probe PRIVATE staging_probe_dll)
set_target_properties(runtime_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/app")
hyle_stage_runtime_dependencies(runtime_probe)
hyle_stage_runtime_dependencies(runtime_probe)
add_executable(empty_probe empty.cc)
set_target_properties(empty_probe PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/empty")
hyle_stage_runtime_dependencies(empty_probe)
]=])
string(CONFIGURE "${_project}" _project @ONLY)
file(WRITE "${_source}/CMakeLists.txt" "${_project}")
file(WRITE "${_source}/library.cc"
  "extern \"C\" __declspec(dllexport) int probe_value() { return PROBE_VALUE; }\n")
file(WRITE "${_source}/main.cc" [=[
#include <cstdlib>
extern "C" __declspec(dllimport) int probe_value();
int main(int argc, char **argv) {
  return argc == 2 && probe_value() == std::atoi(argv[1]) ? 0 : 1;
}
]=])
file(WRITE "${_source}/empty.cc" "int main() { return 0; }\n")

_run("${CMAKE_COMMAND}" -S "${_source}" -B "${_build}" -G Ninja
  -DCMAKE_BUILD_TYPE=Release -DPROBE_VALUE=1)
_run("${CMAKE_COMMAND}" --build "${_build}" --target runtime_probe empty_probe --parallel 2)
set(_exe "${_build}/app/runtime_probe.exe")
set(_dll "${_build}/staging_probe_dll.dll")
set(_staged "${_build}/app/staging_probe_dll.dll")
_run("${_exe}" 1)
_run("${_build}/empty/empty_probe.exe")
file(SHA256 "${_exe}" _exe_before)
file(SHA256 "${_build}/staging_probe_dll.lib" _import_before)
file(SHA256 "${_dll}" _dll_before)

# The public signature and executable do not change, only the DLL's body.
_run("${CMAKE_COMMAND}" -S "${_source}" -B "${_build}" -DPROBE_VALUE=2)
_run("${CMAKE_COMMAND}" --build "${_build}" --target runtime_probe empty_probe --parallel 2)
_run("${_exe}" 2)
file(SHA256 "${_exe}" _exe_after)
file(SHA256 "${_build}/staging_probe_dll.lib" _import_after)
file(SHA256 "${_dll}" _dll_after)
file(SHA256 "${_staged}" _staged_after)
if(NOT _exe_before STREQUAL _exe_after OR NOT _import_before STREQUAL _import_after)
  message(FATAL_ERROR "Fixture did not preserve the EXE/import library identity")
endif()
if(_dll_before STREQUAL _dll_after OR NOT _dll_after STREQUAL _staged_after)
  message(FATAL_ERROR "Changed runtime DLL was not staged")
endif()
file(TIMESTAMP "${_staged}" _staged_time "%s")
_run("${CMAKE_COMMAND}" --build "${_build}" --target runtime_probe empty_probe --parallel 2)
file(TIMESTAMP "${_staged}" _unchanged_time "%s")
if(NOT _staged_time STREQUAL _unchanged_time)
  message(FATAL_ERROR "No-change build rewrote the staged DLL")
endif()
_run("${_exe}" 2)
message(STATUS "Runtime staging PASS: initial/incremental/no-change, stable EXE/import, empty/idempotent")
