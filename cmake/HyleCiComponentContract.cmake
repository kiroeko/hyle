# Configure-only checks for the component-selection and dependency-graph
# contracts.  These deliberately use fresh child projects so cache shadowing
# and source-embedding behavior are exercised exactly as consumers see them.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_SOURCE_DIR HYLE_SMOKE_ROOT)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=<path>")
  endif()
  get_filename_component(${_required} "${${_required}}" ABSOLUTE)
  file(TO_CMAKE_PATH "${${_required}}" ${_required})
endforeach()

set(_hyle_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _hyle_generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()

function(_hyle_contract_run label expect_success)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  if(expect_success AND NOT _result EQUAL 0)
    message(FATAL_ERROR
      "${label} failed with exit code ${_result}:\n${_output}")
  elseif(NOT expect_success AND _result EQUAL 0)
    message(FATAL_ERROR "${label} unexpectedly succeeded")
  endif()
  set(HYLE_CONTRACT_LAST_OUTPUT "${_output}" PARENT_SCOPE)
endfunction()

# An explicitly empty -D value is almost always a lost shell/CI variable.  The
# ordinary unspecified default remains `all`, but explicit empty must fail.
_hyle_contract_run("explicit-empty component selection" FALSE
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
  -B "${HYLE_SMOKE_ROOT}/explicit-empty"
  ${_hyle_generator_args}
  -DHYLE_COMPONENTS:STRING=
  -DHYLE_BUILD_TESTS=OFF
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_IO_TLS=OFF
  -DHYLE_COMPUTE_CUDA=OFF)
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "HYLE_COMPONENTS is explicitly empty")
  message(FATAL_ERROR
    "explicit-empty selection failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# Reproduce the normal-variable form used by add_subdirectory consumers.  The
# generated target graph and the cache/GUI manifest must agree.
set(_embed_source "${HYLE_SMOKE_ROOT}/embed-source")
file(MAKE_DIRECTORY "${_embed_source}")
file(WRITE "${_embed_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_component_embed LANGUAGES C CXX)\n"
"set(HYLE_COMPONENTS container)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_IO_TLS OFF)\n"
"set(HYLE_COMPUTE_CUDA OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"get_property(_cached CACHE HYLE_COMPONENTS PROPERTY VALUE)\n"
"if(NOT _cached STREQUAL \"container\")\n"
"  message(FATAL_ERROR \"cache says '\${_cached}', expected container\")\n"
"endif()\n"
"foreach(_required IN ITEMS error platform mem container)\n"
"  if(NOT TARGET hyle::\${_required})\n"
"    message(FATAL_ERROR \"missing selected-closure target hyle::\${_required}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_forbidden IN ITEMS io compute graph simd)\n"
"  if(TARGET hyle::\${_forbidden})\n"
"    message(FATAL_ERROR \"unexpected target hyle::\${_forbidden}\")\n"
"  endif()\n"
"endforeach()\n")
_hyle_contract_run("source-embedded component/cache agreement" TRUE
  "${CMAKE_COMMAND}" -S "${_embed_source}"
  -B "${HYLE_SMOKE_ROOT}/embed-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# Prove that the end-of-configuration audit catches an internal link added
# after the declaration-time helper has returned.
set(_bypass_source "${HYLE_SMOKE_ROOT}/bypass-source")
file(MAKE_DIRECTORY "${_bypass_source}")
file(WRITE "${_bypass_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_dependency_bypass LANGUAGES NONE)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core error)\n"
"set(_HYLE_DEPS_CORE \"\")\n"
"set(_HYLE_DEPS_ERROR \"\")\n"
"include(HyleModule)\n"
"add_library(hyle_core INTERFACE)\n"
"add_library(hyle_error INTERFACE)\n"
"target_link_libraries(hyle_error INTERFACE hyle_core)\n"
"hyle_validate_final_module_graph()\n")
_hyle_contract_run("late internal dependency bypass" FALSE
  "${CMAKE_COMMAND}" -S "${_bypass_source}"
  -B "${HYLE_SMOKE_ROOT}/bypass-build"
  ${_hyle_generator_args})
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "direct internal dependencies disagree")
  message(FATAL_ERROR
    "late dependency bypass failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# A late PUBLIC edge can also silently widen a package usage contract even when
# the dependency set itself is unchanged.  Prove that visibility is audited,
# not only graph membership.
set(_visibility_source "${HYLE_SMOKE_ROOT}/visibility-source")
file(MAKE_DIRECTORY "${_visibility_source}")
file(WRITE "${_visibility_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_dependency_visibility LANGUAGES NONE)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core error)\n"
"set(_HYLE_DEPS_CORE \"\")\n"
"set(_HYLE_PRIVATE_DEPS_CORE \"\")\n"
"set(_HYLE_PUBLIC_DEPS_CORE \"\")\n"
"set(_HYLE_DEPS_ERROR core)\n"
"set(_HYLE_PRIVATE_DEPS_ERROR core)\n"
"set(_HYLE_PUBLIC_DEPS_ERROR \"\")\n"
"include(HyleModule)\n"
"add_library(hyle_core INTERFACE)\n"
"add_library(hyle_error INTERFACE)\n"
"target_link_libraries(hyle_error INTERFACE hyle_core)\n"
"hyle_validate_final_module_graph()\n")
_hyle_contract_run("late internal dependency visibility widening" FALSE
  "${CMAKE_COMMAND}" -S "${_visibility_source}"
  -B "${HYLE_SMOKE_ROOT}/visibility-build"
  ${_hyle_generator_args})
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "public internal dependencies disagree")
  message(FATAL_ERROR
    "late dependency visibility widening failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

message(STATUS "HYLE component contracts passed")
