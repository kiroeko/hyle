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

# Every generated child is a real configure journey and must inherit the same
# host-required platform floor as the delivery harness. Without this, the Mac
# contract fails while configuring a Platform-dependent child before it can
# evaluate the target assertions it was created to test.
include("${HYLE_SOURCE_DIR}/cmake/HylePlatformRequirements.cmake")
set(_hyle_generator_args "")
if(DEFINED HYLE_SMOKE_GENERATOR AND NOT HYLE_SMOKE_GENERATOR STREQUAL "")
  list(APPEND _hyle_generator_args -G "${HYLE_SMOKE_GENERATOR}")
endif()
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin")
  list(APPEND _hyle_generator_args
    -DCMAKE_OSX_DEPLOYMENT_TARGET=${HYLE_PLATFORM_MACOS_DEPLOYMENT_TARGET_MIN})
endif()

set(_hyle_contract_root "${HYLE_SMOKE_ROOT}/component-contract")

file(MAKE_DIRECTORY "${_hyle_contract_root}")

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

set(_hyle_direct_public_edge_helper
  "${_hyle_contract_root}/HyleRequireDirectPublicEdge.cmake")
file(WRITE "${_hyle_direct_public_edge_helper}" [=[
function(hyle_contract_require_direct_public_edge consumer dependency)
  if(NOT TARGET "${consumer}")
    message(FATAL_ERROR
      "cannot inspect missing consumer target '${consumer}'")
  endif()
  if(NOT TARGET "${dependency}")
    message(FATAL_ERROR
      "cannot inspect missing dependency target '${dependency}'")
  endif()
  get_target_property(
    _hyle_direct_public_links "${consumer}" INTERFACE_LINK_LIBRARIES)
  list(FIND _hyle_direct_public_links "${dependency}"
    _hyle_direct_public_index)
  if(_hyle_direct_public_index EQUAL -1)
    message(FATAL_ERROR
      "target '${consumer}' has no direct PUBLIC edge to '${dependency}'; "
      "INTERFACE_LINK_LIBRARIES='${_hyle_direct_public_links}'")
  endif()
endfunction()
function(hyle_contract_require_direct_private_edge consumer dependency)
  if(NOT TARGET "${consumer}" OR NOT TARGET "${dependency}")
    message(FATAL_ERROR
      "cannot inspect PRIVATE edge '${consumer}' -> '${dependency}'")
  endif()
  get_target_property(_hyle_direct_links "${consumer}" LINK_LIBRARIES)
  set(_hyle_direct_found FALSE)
  foreach(_hyle_direct_link IN LISTS _hyle_direct_links)
    if(_hyle_direct_link MATCHES
       "(^|[^A-Za-z0-9_])${dependency}([^A-Za-z0-9_]|$)")
      set(_hyle_direct_found TRUE)
      break()
    endif()
  endforeach()
  if(NOT _hyle_direct_found)
    message(FATAL_ERROR
      "target '${consumer}' has no direct PRIVATE edge to '${dependency}'; "
      "LINK_LIBRARIES='${_hyle_direct_links}'")
  endif()
endfunction()
]=])

# Maintainer `all` selects every implemented protocol/workflow factor.
_hyle_contract_run("maintainer all selects every implemented IO factor" TRUE
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
  -B "${_hyle_contract_root}/implemented-factors-all"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DHYLE_DEV_FACTORS=all
  -DHYLE_BUILD_TESTS=OFF
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_INSTALL=OFF
  -DHYLE_IO_TLS=OFF
  -DHYLE_IO_HTTP2=OFF)
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "resolved='[^']*io_tls[^']*io_http[^']*io_http2[^']*io_transfer")
  message(FATAL_ERROR
    "maintainer all did not select the complete implemented IO factor set:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

set(_io_transfer_factor_build
  "${_hyle_contract_root}/io-transfer-factor")
_hyle_contract_run("exact io_transfer factor is selectable" TRUE
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
  -B "${_io_transfer_factor_build}"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DHYLE_DEV_FACTORS=io_transfer
  -DHYLE_BUILD_TESTS=ON
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_IO_TLS=OFF
  -DHYLE_IO_HTTP2=OFF
  -DHYLE_INSTALL=OFF)
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "resolved='[^']*io_core[^']*io_tls[^']*io_http[^']*io_transfer")
  message(FATAL_ERROR
    "exact io_transfer factor resolved the wrong dependency closure:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()
_hyle_contract_run("build exact io_transfer factor tests" TRUE
  "${CMAKE_COMMAND}" --build "${_io_transfer_factor_build}"
  --config Release --target hyle_io_transfer_tests --parallel 1)
unset(_io_transfer_factor_build)

# Component identity order is append-only audit history, not a dependency
# schedule. Obs predates the later mem_reclaim factor in that ledger but may
# still acquire it as a natural private dependency. Prove that the selected
# closure is constructed in stable topological order without moving either ID.
set(_append_only_topology_source
  "${_hyle_contract_root}/append-only-topology-source")
file(MAKE_DIRECTORY "${_append_only_topology_source}")
file(WRITE "${_append_only_topology_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_append_only_topology LANGUAGES CXX)\n"
"set(HYLE_COMPONENTS obs)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"set(_expected core error mem_reclaim time obs)\n"
"get_directory_property(_resolved DIRECTORY \"\${CMAKE_CURRENT_BINARY_DIR}/hyle\" DEFINITION HYLE_RESOLVED_COMPONENT_LIST)\n"
"if(NOT _resolved STREQUAL _expected)\n"
"  message(FATAL_ERROR \"obs topology is '\${_resolved}'\")\n"
"endif()\n"
"foreach(_factor IN LISTS _expected)\n"
"  if(NOT TARGET hyle_\${_factor})\n"
"    message(FATAL_ERROR \"missing topological target hyle_\${_factor}\")\n"
"  endif()\n"
"endforeach()\n")
_hyle_contract_run("append-only identity has topological construction" TRUE
  "${CMAKE_COMMAND}" -S "${_append_only_topology_source}"
  -B "${_hyle_contract_root}/append-only-topology-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)
unset(_append_only_topology_source)

# The user-facing Time product aggregates a host-clock factor and the
# header-only timer_queue factor. The host artifact itself stays independent
# from intrusive storage, while the timer factor declares every header edge.
set(_time_source "${_hyle_contract_root}/time-source")
file(MAKE_DIRECTORY "${_time_source}")
file(WRITE "${_time_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_time_component LANGUAGES CXX)\n"
"set(HYLE_COMPONENTS time)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"include(\"${_hyle_direct_public_edge_helper}\")\n"
"foreach(_required_product IN ITEMS core time)\n"
"  if(NOT TARGET hyle::\${_required_product})\n"
"    message(FATAL_ERROR \"missing time product hyle::\${_required_product}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_required_factor IN ITEMS core container_intrusive time timer_queue)\n"
"  if(NOT TARGET hyle_\${_required_factor})\n"
"    message(FATAL_ERROR \"missing time factor hyle_\${_required_factor}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_internal IN ITEMS container_intrusive timer_queue)\n"
"  if(TARGET hyle::\${_internal})\n"
"    message(FATAL_ERROR \"internal factor leaked as product hyle::\${_internal}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_forbidden IN ITEMS mem platform container)\n"
"  if(TARGET hyle::\${_forbidden})\n"
"    message(FATAL_ERROR \"unexpected time-closure target hyle::\${_forbidden}\")\n"
"  endif()\n"
"endforeach()\n"
"hyle_contract_require_direct_public_edge(hyle_time hyle_core)\n"
"hyle_contract_require_direct_public_edge(hyle_timer_queue hyle_core)\n"
"hyle_contract_require_direct_public_edge(hyle_timer_queue hyle_time)\n"
"hyle_contract_require_direct_public_edge(hyle_timer_queue hyle_container_intrusive)\n")
_hyle_contract_run("time separates host and timer factors" TRUE
  "${CMAKE_COMMAND}" -S "${_time_source}"
  -B "${_hyle_contract_root}/time-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# Reproduce the normal-variable form used by add_subdirectory consumers.  The
# generated target graph and the cache/GUI manifest must agree.
set(_embed_source "${_hyle_contract_root}/embed-source")
file(MAKE_DIRECTORY "${_embed_source}")
file(WRITE "${_embed_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_component_embed LANGUAGES CXX)\n"
"if(APPLE)\n"
"  set(CMAKE_OSX_DEPLOYMENT_TARGET 14.4 CACHE STRING \"\" FORCE)\n"
"endif()\n"
"set(HYLE_COMPONENTS container)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_IO_TLS OFF)\n"
"set(HYLE_SANITIZER NONE)\n"
"set(HYLE_DEP_SOURCE_CACHE \"${HYLE_SMOKE_ROOT}/shared-source-cache\")\n"
"set(HYLE_DEP_BUILD_DIR \"\${CMAKE_BINARY_DIR}/hyle-dependencies\")\n"
"set(FETCHCONTENT_BASE_DIR \"\${CMAKE_BINARY_DIR}/parent-dependencies\")\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"include(\"${_hyle_direct_public_edge_helper}\")\n"
"if(NOT FETCHCONTENT_BASE_DIR STREQUAL \"\${CMAKE_BINARY_DIR}/parent-dependencies\")\n"
"  message(FATAL_ERROR \"HYLE rewrote its embedding parent's FetchContent cache\")\n"
"endif()\n"
"get_property(_cached CACHE HYLE_COMPONENTS PROPERTY VALUE)\n"
"if(NOT _cached STREQUAL \"container\")\n"
"  message(FATAL_ERROR \"cache says '\${_cached}', expected container\")\n"
"endif()\n"
"get_property(_deps_cached CACHE HYLE_DEP_BUILD_DIR PROPERTY VALUE)\n"
"get_directory_property(_deps_effective DIRECTORY \"\${CMAKE_CURRENT_BINARY_DIR}/hyle\" DEFINITION HYLE_DEP_BUILD_DIR)\n"
"if(NOT _deps_cached STREQUAL \"\${CMAKE_BINARY_DIR}/hyle-dependencies\" OR\n"
"   NOT _deps_effective STREQUAL \"\${CMAKE_BINARY_DIR}/hyle-dependencies\")\n"
"  message(FATAL_ERROR \"HYLE_DEP_BUILD_DIR cache/effective split: cache='\${_deps_cached}', effective='\${_deps_effective}'\")\n"
"endif()\n"
"get_property(_source_cache_cached CACHE HYLE_DEP_SOURCE_CACHE PROPERTY VALUE)\n"
"get_directory_property(_source_cache_effective DIRECTORY \"\${CMAKE_CURRENT_BINARY_DIR}/hyle\" DEFINITION HYLE_DEP_SOURCE_CACHE)\n"
"if(NOT _source_cache_cached STREQUAL \"${HYLE_SMOKE_ROOT}/shared-source-cache\" OR\n"
"   NOT _source_cache_effective STREQUAL \"${HYLE_SMOKE_ROOT}/shared-source-cache\")\n"
"  message(FATAL_ERROR \"HYLE_DEP_SOURCE_CACHE cache/effective split: cache='\${_source_cache_cached}', effective='\${_source_cache_effective}'\")\n"
"endif()\n"
"get_property(_sanitizer_cached CACHE HYLE_SANITIZER PROPERTY VALUE)\n"
"get_directory_property(_sanitizer_effective DIRECTORY \"\${CMAKE_CURRENT_BINARY_DIR}/hyle\" DEFINITION HYLE_SANITIZER)\n"
"if(NOT _sanitizer_cached STREQUAL \"off\" OR NOT _sanitizer_effective STREQUAL \"off\")\n"
"  message(FATAL_ERROR \"HYLE_SANITIZER cache/effective split: cache='\${_sanitizer_cached}', effective='\${_sanitizer_effective}'\")\n"
"endif()\n"
"foreach(_required_product IN ITEMS core error container)\n"
"  if(NOT TARGET hyle::\${_required_product})\n"
"    message(FATAL_ERROR \"missing selected product hyle::\${_required_product}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_required_factor IN ITEMS core container_intrusive error platform mem_reclaim mem container container_concurrent container_queue container_text)\n"
"  if(NOT TARGET hyle_\${_required_factor})\n"
"    message(FATAL_ERROR \"missing selected factor hyle_\${_required_factor}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_internal_alias IN ITEMS container_intrusive container_concurrent container_queue container_text util_core)\n"
"  if(TARGET hyle::\${_internal_alias})\n"
"    message(FATAL_ERROR \"internal factor leaked as hyle::\${_internal_alias}\")\n"
"  endif()\n"
"endforeach()\n"
"foreach(_forbidden IN ITEMS time platform mem obs simd math util sched io compute graph)\n"
"  if(TARGET hyle::\${_forbidden})\n"
"    message(FATAL_ERROR \"unexpected target hyle::\${_forbidden}\")\n"
"  endif()\n"
"endforeach()\n"
"hyle_contract_require_direct_public_edge(hyle_container hyle_core)\n"
"hyle_contract_require_direct_public_edge(hyle_container hyle_mem)\n"
"hyle_contract_require_direct_public_edge(hyle_container hyle_container_intrusive)\n"
"hyle_contract_require_direct_public_edge(hyle_container_concurrent hyle_core)\n"
"hyle_contract_require_direct_public_edge(hyle_container_concurrent hyle_mem)\n"
"hyle_contract_require_direct_public_edge(hyle_container_concurrent hyle_mem_reclaim)\n"
"hyle_contract_require_direct_public_edge(hyle_container_concurrent hyle_container)\n"
"hyle_contract_require_direct_public_edge(hyle_container_queue hyle_core)\n"
"hyle_contract_require_direct_public_edge(hyle_container_queue hyle_mem)\n"
"hyle_contract_require_direct_public_edge(hyle_container_text hyle_core)\n"
"hyle_contract_require_direct_public_edge(hyle_container_text hyle_mem)\n"
"hyle_contract_require_direct_public_edge(hyle_container_text hyle_container)\n"
"get_target_property(_global_build_options hyle_build_contract INTERFACE_COMPILE_OPTIONS)\n"
"if(_global_build_options MATCHES \"mssse3|/EHsc\")\n"
"  message(FATAL_ERROR \"global build contract leaked product policy: \${_global_build_options}\")\n"
"endif()\n"
"get_property(_hyle_processor GLOBAL PROPERTY HYLE_TARGET_PROCESSOR)\n"
"get_target_property(_container_options hyle_container INTERFACE_COMPILE_OPTIONS)\n"
"if(_container_options MATCHES \"mssse3|SSSE3\")\n"
"  message(FATAL_ERROR \"Container leaked a higher x86 ISA into consumer TUs: \${_container_options}\")\n"
"endif()\n")
_hyle_contract_run("source-embedded component/cache agreement" TRUE
  "${CMAKE_COMMAND}" -S "${_embed_source}"
  -B "${_hyle_contract_root}/embed-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)
_hyle_contract_run("source-embedded disconnected cache reuse" TRUE
  "${CMAKE_COMMAND}" -S "${_embed_source}"
  -B "${_hyle_contract_root}/embed-build-disconnected"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON)

# The embedding fixture above populated one real central-lock identity in an
# isolated shared source cache. Reuse that exact archive to exercise locking,
# corruption handling, offline repair, and guarded stale pruning without a
# second network download or a synthetic dependency ledger.
_hyle_contract_run("shared dependency source-cache lifecycle" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -DHYLE_SEED_SOURCE_CACHE=${HYLE_SMOKE_ROOT}/shared-source-cache
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiDependencyCacheContract.cmake")

# Exact-ISA spelling is one shared input to SIMD codegen and optional GGML
# configuration. Normalize a parent normal variable once so those consumers
# cannot silently interpret lowercase values differently.
set(_simd_exact_source "${_hyle_contract_root}/simd-exact-source")
file(MAKE_DIRECTORY "${_simd_exact_source}")
file(WRITE "${_simd_exact_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_simd_exact_contract LANGUAGES CXX)\n"
"set(HYLE_COMPONENTS simd)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"set(HYLE_SIMD_EXACT_TARGET scalar)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"get_property(_exact_cached CACHE HYLE_SIMD_EXACT_TARGET PROPERTY VALUE)\n"
"get_directory_property(_exact_effective DIRECTORY \"\${CMAKE_CURRENT_BINARY_DIR}/hyle\" DEFINITION HYLE_SIMD_EXACT_TARGET)\n"
"if(NOT _exact_cached STREQUAL \"SCALAR\" OR NOT _exact_effective STREQUAL \"SCALAR\")\n"
"  message(FATAL_ERROR \"SIMD exact target cache/effective split: cache='\${_exact_cached}', effective='\${_exact_effective}'\")\n"
"endif()\n")
_hyle_contract_run("SIMD exact target canonicalization" TRUE
  "${CMAKE_COMMAND}" -S "${_simd_exact_source}"
  -B "${_hyle_contract_root}/simd-exact-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# On a host with HYLE's fcontext implementation, io exposes coroutine
# transport backed by sched/time.  Assert the public edges themselves rather
# than only the closure derived from the same component manifest.
set(_transport_source "${_hyle_contract_root}/transport-source")
file(MAKE_DIRECTORY "${_transport_source}")
file(WRITE "${_transport_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_transport_component LANGUAGES CXX)\n"
"if(APPLE)\n"
"  set(CMAKE_OSX_DEPLOYMENT_TARGET 14.4 CACHE STRING \"\" FORCE)\n"
"endif()\n"
"set(HYLE_COMPONENTS io)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"set(HYLE_IO_TLS OFF)\n"
"set(HYLE_IO_HTTP2 OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"include(\"${_hyle_direct_public_edge_helper}\")\n"
"get_directory_property(_hyle_fcontext_supported\n"
"  DIRECTORY \"\${CMAKE_CURRENT_BINARY_DIR}/hyle\"\n"
"  DEFINITION HYLE_FCONTEXT_SUPPORTED)\n"
"if(NOT _hyle_fcontext_supported)\n"
"  message(FATAL_ERROR \"test host unexpectedly lacks supported fcontext\")\n"
"endif()\n"
"foreach(_required_product IN ITEMS time sched io)\n"
"  if(NOT TARGET hyle::\${_required_product})\n"
"    message(FATAL_ERROR \"missing transport product hyle::\${_required_product}\")\n"
"  endif()\n"
"endforeach()\n"
"if(TARGET hyle::container)\n"
"  message(FATAL_ERROR \"transport closure manufactured a partial Container facade\")\n"
"endif()\n"
"if(NOT TARGET hyle_container_intrusive OR TARGET hyle::container_intrusive)\n"
"  message(FATAL_ERROR \"transport factor/product projection is invalid\")\n"
"endif()\n"
"hyle_contract_require_direct_public_edge(hyle_io_core hyle_time)\n"
"hyle_contract_require_direct_public_edge(hyle_io_core hyle_timer_queue)\n"
"hyle_contract_require_direct_public_edge(hyle_io_core hyle_sched)\n"
"hyle_contract_require_direct_public_edge(hyle_sched hyle_container_intrusive)\n")
_hyle_contract_run("supported transport direct PUBLIC edges" TRUE
  "${CMAKE_COMMAND}" -S "${_transport_source}"
  -B "${_hyle_contract_root}/transport-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# Prove that the end-of-configuration audit catches an internal link added
# after the declaration-time helper has returned.
set(_bypass_source "${_hyle_contract_root}/bypass-source")
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
  -B "${_hyle_contract_root}/bypass-build"
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
set(_visibility_source "${_hyle_contract_root}/visibility-source")
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
  -B "${_hyle_contract_root}/visibility-build"
  ${_hyle_generator_args})
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "public internal dependencies disagree")
  message(FATAL_ERROR
    "late dependency visibility widening failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# FetchContent documents that the first declaration of a dependency name wins
# across the complete project hierarchy, while its standard SOURCE_DIR
# override has higher precedence. HYLE's shared-cache resolver deliberately
# installs that authenticated override: a same-named arbitrary parent
# declaration must neither execute nor replace the locked product source.
set(_first_declare_source
  "${_hyle_contract_root}/fetchcontent-first-declare-source")
set(_first_declare_parent
  "${_hyle_contract_root}/fetchcontent-first-declare-parent")
file(MAKE_DIRECTORY "${_first_declare_source}" "${_first_declare_parent}")
file(WRITE "${_first_declare_source}/LICENSE"
  "not the locked GoogleTest license\n")
file(WRITE "${_first_declare_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(not_locked_googletest LANGUAGES NONE)\n"
"message(FATAL_ERROR \"arbitrary parent-declared source executed\")\n")
file(WRITE "${_first_declare_parent}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_fetchcontent_first_declare LANGUAGES CXX)\n"
"include(FetchContent)\n"
"FetchContent_Declare(googletest SOURCE_DIR \"${_first_declare_source}\")\n"
"set(HYLE_COMPONENTS core)\n"
"set(HYLE_BUILD_TESTS ON)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"set(HYLE_IO_TLS OFF)\n"
"set(HYLE_IO_HTTP2 OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n")
_hyle_contract_run("locked source overrides arbitrary first declaration" TRUE
  "${CMAKE_COMMAND}" -S "${_first_declare_parent}"
  -B "${_hyle_contract_root}/fetchcontent-first-declare-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# A source-embedding parent may already own the canonical GoogleTest target.
# HYLE must reuse it instead of populating a second project whose unnamespaced
# gtest/gtest_main targets collide with the parent's copy.
set(_parent_gtest_source
  "${_hyle_contract_root}/parent-gtest-source")
file(MAKE_DIRECTORY "${_parent_gtest_source}")
file(WRITE "${_parent_gtest_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_parent_gtest_contract LANGUAGES CXX)\n"
"add_library(parent_gtest_main INTERFACE)\n"
"add_library(GTest::gtest_main ALIAS parent_gtest_main)\n"
"set(HYLE_COMPONENTS core)\n"
"set(HYLE_BUILD_TESTS ON)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"if(TARGET gtest OR TARGET gtest_main)\n"
"  message(FATAL_ERROR \"HYLE populated a second GoogleTest target set\")\n"
"endif()\n")
_hyle_contract_run("reuse parent-provided GoogleTest target" TRUE
  "${CMAKE_COMMAND}" -S "${_parent_gtest_source}"
  -B "${_hyle_contract_root}/parent-gtest-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

# Lock projection must not require an unused in-tree snapshot.  A selected
# dependency and the complete-audit script entry point must still fail closed.
set(_vendor_fixture "${_hyle_contract_root}/vendor-closure-fixture")
file(MAKE_DIRECTORY "${_vendor_fixture}/cmake")
configure_file("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake"
  "${_vendor_fixture}/cmake/HyleDependencyVersions.cmake" COPYONLY)
file(WRITE "${_vendor_fixture}/cmake/HyleDependencyLock.json"
"{
  \"schema_version\": 3,
  \"expanded_tree_algorithm\": \"hyle-path-hex-v1-raw-sha256\",
  \"dependencies\": [{
    \"name\": \"demo\", \"class\": \"in_tree\",
    \"repository\": \"https://github.com/example/demo.git\",
    \"version\": \"fixture\",
    \"revision\": \"1111111111111111111111111111111111111111\",
    \"git_tree\": \"2222222222222222222222222222222222222222\",
    \"archive\": {
      \"url\": \"https://codeload.github.com/example/demo/tar.gz/1111111111111111111111111111111111111111\",
      \"sha256\": \"3333333333333333333333333333333333333333333333333333333333333333\"
    },
    \"vendored\": {
      \"path\": \"third_party/demo\",
      \"tree_sha256\": \"4444444444444444444444444444444444444444444444444444444444444444\"
    },
    \"license\": { \"expression\": \"MIT\", \"files\": [{
      \"root\": \"hyle\", \"path\": \"third_party/demo/LICENSE\",
      \"sha256\": \"5555555555555555555555555555555555555555555555555555555555555555\"
    }] }
  }]
}
")
set(_vendor_unused "${_vendor_fixture}/unused.cmake")
file(WRITE "${_vendor_unused}"
"cmake_minimum_required(VERSION 3.25)\n"
"include(\"${_vendor_fixture}/cmake/HyleDependencyVersions.cmake\")\n"
"hyle_audit_in_tree_dependencies()\n")
_hyle_contract_run("unused in-tree dependency is not required" TRUE
  "${CMAKE_COMMAND}" -P "${_vendor_unused}")

set(_vendor_selected "${_vendor_fixture}/selected.cmake")
file(WRITE "${_vendor_selected}"
"cmake_minimum_required(VERSION 3.25)\n"
"include(\"${_vendor_fixture}/cmake/HyleDependencyVersions.cmake\")\n"
"hyle_audit_in_tree_dependencies(demo)\n")
_hyle_contract_run("selected in-tree dependency is required" FALSE
  "${CMAKE_COMMAND}" -P "${_vendor_selected}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "required by the selected closure is missing")
  message(FATAL_ERROR
    "selected vendor failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

_hyle_contract_run("complete in-tree audit requires every snapshot" FALSE
  "${CMAKE_COMMAND}" -P
  "${_vendor_fixture}/cmake/HyleDependencyVersions.cmake")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "required by the selected closure is missing")
  message(FATAL_ERROR
    "complete vendor audit failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# License roots are class semantics, not an optional verifier hint.  A bad
# root must fail during projection even when that dependency is not selected;
# otherwise one audit path could silently skip the license entirely.
file(READ "${_vendor_fixture}/cmake/HyleDependencyLock.json"
  _vendor_lock_json)
string(REPLACE "\"root\": \"hyle\"" "\"root\": \"dependency\""
  _vendor_bad_root_json "${_vendor_lock_json}")
set(_vendor_bad_root_lock "${_vendor_fixture}/bad-license-root.json")
file(WRITE "${_vendor_bad_root_lock}" "${_vendor_bad_root_json}")
set(_vendor_bad_root_script "${_vendor_fixture}/bad-license-root.cmake")
file(WRITE "${_vendor_bad_root_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_vendor_bad_root_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_vendor_bad_root_lock}\")\n"
"include(\"${_vendor_fixture}/cmake/HyleDependencyVersions.cmake\")\n")
_hyle_contract_run("class-specific license root" FALSE
  "${CMAKE_COMMAND}" -P "${_vendor_bad_root_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "class 'in_tree' requires license root 'hyle'")
  message(FATAL_ERROR
    "bad license root failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

string(REPLACE
  "third_party/demo/LICENSE" "third_party/other/LICENSE"
  _vendor_bad_owner_json "${_vendor_lock_json}")
set(_vendor_bad_owner_lock "${_vendor_fixture}/bad-license-owner.json")
file(WRITE "${_vendor_bad_owner_lock}" "${_vendor_bad_owner_json}")
set(_vendor_bad_owner_script "${_vendor_fixture}/bad-license-owner.cmake")
file(WRITE "${_vendor_bad_owner_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_vendor_bad_owner_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_vendor_bad_owner_lock}\")\n"
"include(\"${_vendor_fixture}/cmake/HyleDependencyVersions.cmake\")\n")
_hyle_contract_run("in-tree license ownership" FALSE
  "${CMAKE_COMMAND}" -P "${_vendor_bad_owner_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "outside its vendored path")
  message(FATAL_ERROR
    "cross-vendor license path failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

file(READ "${HYLE_SOURCE_DIR}/cmake/HyleDependencyLock.json"
  _main_dependency_lock_json)
string(JSON _main_dependency_count LENGTH
  "${_main_dependency_lock_json}" dependencies)
math(EXPR _main_dependency_last "${_main_dependency_count} - 1")
unset(_highway_port_index)
foreach(_main_dependency_index RANGE 0 ${_main_dependency_last})
  string(JSON _main_dependency_name GET
    "${_main_dependency_lock_json}"
    dependencies ${_main_dependency_index} name)
  if(_main_dependency_name STREQUAL "highway_port")
    set(_highway_port_index ${_main_dependency_index})
  endif()
endforeach()
if(NOT DEFINED _highway_port_index)
  message(FATAL_ERROR "central lock has no highway_port integrated source")
endif()
string(REPLACE
  "third_party/ported/LICENSE-HIGHWAY"
  "third_party/foreign/LICENSE-HIGHWAY"
  _integrated_bad_owner_json "${_main_dependency_lock_json}")
set(_integrated_bad_owner_lock
  "${_hyle_contract_root}/integrated-bad-license-owner.json")
file(WRITE "${_integrated_bad_owner_lock}" "${_integrated_bad_owner_json}")
set(_integrated_bad_owner_script
  "${_hyle_contract_root}/integrated-bad-license-owner.cmake")
file(WRITE "${_integrated_bad_owner_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_integrated_bad_owner_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_integrated_bad_owner_lock}\")\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake\")\n")

# A command-line cache variable is data, not an authority to replace the
# central ledger. Trusted fixture code can attest one exact alternate path via
# a GLOBAL property, but a plain -D override must be rejected before parsing.
set(_untrusted_lock_override_script
  "${_hyle_contract_root}/untrusted-lock-override.cmake")
file(WRITE "${_untrusted_lock_override_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake\")\n")
_hyle_contract_run("untrusted central lock override" FALSE
  "${CMAKE_COMMAND}"
  -DHYLE_DEPENDENCY_LOCK_FILE=${_integrated_bad_owner_lock}
  -P "${_untrusted_lock_override_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "Refusing untrusted HYLE_DEPENDENCY_LOCK_FILE override")
  message(FATAL_ERROR
    "untrusted central lock override failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()
_hyle_contract_run("integrated-source license ownership" FALSE
  "${CMAKE_COMMAND}" -P "${_integrated_bad_owner_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "must be under 'third_party/ported/'")
  message(FATAL_ERROR
    "integrated license ownership failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

string(JSON _integrated_bad_content_json SET
  "${_main_dependency_lock_json}"
  dependencies ${_highway_port_index} integrated content_sha256
  "\"0000000000000000000000000000000000000000000000000000000000000000\"")
set(_integrated_bad_content_lock
  "${_hyle_contract_root}/integrated-bad-content.json")
file(WRITE "${_integrated_bad_content_lock}" "${_integrated_bad_content_json}")
set(_integrated_bad_content_script
  "${_hyle_contract_root}/integrated-bad-content.cmake")
file(WRITE "${_integrated_bad_content_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_integrated_bad_content_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_integrated_bad_content_lock}\")\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake\")\n"
"hyle_audit_integrated_source_dependencies(highway_port)\n")
_hyle_contract_run("integrated-source adapted content" FALSE
  "${CMAKE_COMMAND}" -P "${_integrated_bad_content_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES
   "adapted content does not match the lock")
  message(FATAL_ERROR
    "integrated content mismatch failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# Repeating an adapted path would make the hash projection ambiguous: one lock
# could appear to cover more surfaces while hashing the same bytes twice.
string(JSON _highway_integrated_path_count LENGTH
  "${_main_dependency_lock_json}"
  dependencies ${_highway_port_index} integrated paths)
string(JSON _highway_first_integrated_path GET
  "${_main_dependency_lock_json}"
  dependencies ${_highway_port_index} integrated paths 0)
string(JSON _integrated_duplicate_path_json SET
  "${_main_dependency_lock_json}"
  dependencies ${_highway_port_index} integrated paths
  ${_highway_integrated_path_count}
  "\"${_highway_first_integrated_path}\"")
set(_integrated_duplicate_path_lock
  "${_hyle_contract_root}/integrated-duplicate-path.json")
file(WRITE "${_integrated_duplicate_path_lock}"
  "${_integrated_duplicate_path_json}")
set(_integrated_duplicate_path_script
  "${_hyle_contract_root}/integrated-duplicate-path.cmake")
file(WRITE "${_integrated_duplicate_path_script}"
"cmake_minimum_required(VERSION 3.25)\n"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_integrated_duplicate_path_lock}\")\n"
"set(HYLE_DEPENDENCY_LOCK_FILE \"${_integrated_duplicate_path_lock}\")\n"
"include(\"${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake\")\n")
_hyle_contract_run("integrated-source duplicate path" FALSE
  "${CMAKE_COMMAND}" -P "${_integrated_duplicate_path_script}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "has duplicate HYLE path")
  message(FATAL_ERROR
    "integrated duplicate path failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

# External implementation/package facts also have one owner and one
# declaration.  Exercise the registry in script mode so a duplicate name or a
# stale exported target cannot degrade into a partially relocatable SDK.
set(_package_registry_projection
  "${_hyle_contract_root}/package-registry-projection.cmake")
file(WRITE "${_package_registry_projection}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"set(HYLE_STATIC ON)\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation FEATURES demo_feature\n"
"  FIND_DEPENDENCIES CUDAToolkit|13.3\n"
"  STATIC_FIND_DEPENDENCIES Threads OpenSSL|3.0)\n"
"hyle_package_collect_registry(_targets _features _find_dependencies)\n"
"if(_targets OR NOT _features STREQUAL \"demo_feature\" OR\n"
"   NOT _find_dependencies STREQUAL \"CUDAToolkit|13.3;Threads;OpenSSL|3.0\")\n"
"  message(FATAL_ERROR \"unexpected registry projection\")\n"
"endif()\n")
_hyle_contract_run("package dependency registry projection" TRUE
  "${CMAKE_COMMAND}" -P "${_package_registry_projection}")

set(_package_registry_duplicate
  "${_hyle_contract_root}/package-registry-duplicate.cmake")
file(WRITE "${_package_registry_duplicate}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(NAME demo OWNER foundation)\n"
"hyle_register_package_dependency(NAME demo OWNER foundation)\n")
_hyle_contract_run("duplicate package dependency registration" FALSE
  "${CMAKE_COMMAND}" -P "${_package_registry_duplicate}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "registered more than once")
  message(FATAL_ERROR
    "duplicate package dependency failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

set(_package_registry_missing_target
  "${_hyle_contract_root}/package-registry-missing-target.cmake")
file(WRITE "${_package_registry_missing_target}"
"cmake_minimum_required(VERSION 3.25)\n"
"list(APPEND CMAKE_MODULE_PATH \"${HYLE_SOURCE_DIR}/cmake\")\n"
"set(HYLE_KNOWN_COMPONENTS core)\n"
"set(HYLE_STATIC ON)\n"
"include(HylePackageRegistry)\n"
"hyle_register_package_dependency(\n"
"  NAME demo OWNER foundation STATIC_EXPORT_TARGETS absent_vendor)\n"
"hyle_package_collect_registry(_targets _features _find_dependencies)\n")
_hyle_contract_run("missing package dependency export target" FALSE
  "${CMAKE_COMMAND}" -P "${_package_registry_missing_target}")
if(NOT HYLE_CONTRACT_LAST_OUTPUT MATCHES "requires missing export target")
  message(FATAL_ERROR
    "missing package dependency target failed for the wrong reason:\n"
    "${HYLE_CONTRACT_LAST_OUTPUT}")
endif()

_hyle_contract_run("locked package/license registry contract" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiPackageRegistryContract.cmake")

_hyle_contract_run("64-bit target-platform contract" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -DHYLE_SMOKE_GENERATOR=${HYLE_SMOKE_GENERATOR}
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiTargetPlatformContract.cmake")

# Exercise the strict offline preload in a self-contained local Git fixture:
# allow-listing, HEAD/tree/clean checks, license hashes, no-network cache gates,
# and the explicit non-Git escape hatch all have positive/negative coverage.
_hyle_contract_run("verified offline dependency contract" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_SOURCE_DIR=${HYLE_SOURCE_DIR}
  -DHYLE_SMOKE_ROOT=${_hyle_contract_root}
  -DHYLE_SMOKE_GENERATOR=${HYLE_SMOKE_GENERATOR}
  -P "${HYLE_SOURCE_DIR}/cmake/HyleCiOfflineDependencyContract.cmake")

# HYLE_DEP_BUILD_DIR is the one build-local binary/subbuild location for every
# HYLE-owned FetchContent dependency. Exercise the test-only path through a self-locked
# GoogleTest-shaped fixture; arbitrary fake direct overrides are intentionally
# rejected by the production provenance wrapper.
set(_fake_gtest_source "${_hyle_contract_root}/fake-googletest")
set(_fake_gtest_lock "${_hyle_contract_root}/fake-googletest-lock.json")
set(_error_test_build "${_hyle_contract_root}/error-test-build")
set(_error_dependency_cache "${_hyle_contract_root}/error-dependencies")
file(MAKE_DIRECTORY "${_fake_gtest_source}")
file(WRITE "${_fake_gtest_source}/LICENSE" "fixture license\n")
file(WRITE "${_fake_gtest_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(fake_googletest LANGUAGES CXX)\n"
"option(BUILD_GMOCK \"fixture gmock switch\" ON)\n"
"option(INSTALL_GTEST \"fixture install switch\" ON)\n"
"option(GTEST_HAS_ABSL \"fixture Abseil switch\" ON)\n"
"option(gtest_build_samples \"fixture sample switch\" ON)\n"
"option(gtest_build_tests \"fixture self-test switch\" ON)\n"
"if(BUILD_GMOCK OR INSTALL_GTEST OR GTEST_HAS_ABSL OR gtest_build_samples OR gtest_build_tests)\n"
"  message(FATAL_ERROR \"HYLE did not suppress GoogleTest optional products\")\n"
"endif()\n"
"add_library(gtest INTERFACE)\n"
"add_library(GTest::gtest ALIAS gtest)\n"
"add_library(gtest_main INTERFACE)\n"
"add_library(GTest::gtest_main ALIAS gtest_main)\n")
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake")
hyle_compute_expanded_source_tree_sha256(
  "${_fake_gtest_source}" _fake_gtest_expanded_tree)
string(SHA256 _fake_gtest_license_sha256 "fixture license\n")
file(WRITE "${_fake_gtest_lock}"
"{
  \"schema_version\": 3,
  \"expanded_tree_algorithm\": \"hyle-path-hex-v1-raw-sha256\",
  \"dependencies\": [{
    \"name\": \"googletest\", \"class\": \"fetchcontent\",
    \"repository\": \"${HYLE_DEP_GOOGLETEST_REPOSITORY}\",
    \"version\": \"fixture-${HYLE_DEP_GOOGLETEST_VERSION}\",
    \"revision\": \"${HYLE_DEP_GOOGLETEST_COMMIT}\",
    \"git_tree\": \"${HYLE_DEP_GOOGLETEST_GIT_TREE}\",
    \"archive\": {
      \"url\": \"${HYLE_DEP_GOOGLETEST_ARCHIVE_URL}\",
      \"sha256\": \"${HYLE_DEP_GOOGLETEST_ARCHIVE_SHA256}\"
    },
    \"expanded_tree_sha256\": \"${_fake_gtest_expanded_tree}\",
    \"license\": { \"expression\": \"BSD-3-Clause\", \"files\": [{
      \"root\": \"dependency\", \"path\": \"LICENSE\",
      \"sha256\": \"${_fake_gtest_license_sha256}\"
    }] }
  }]
}
")
set(_fake_gtest_lock_attestation
  "${_hyle_contract_root}/fake-gtest-lock-attestation.cmake")
file(WRITE "${_fake_gtest_lock_attestation}"
"set_property(GLOBAL PROPERTY HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE \"${_fake_gtest_lock}\")\n")
_hyle_contract_run("error-only dependency cache ownership" TRUE
  "${CMAKE_COMMAND}" -S "${HYLE_SOURCE_DIR}"
  -B "${_error_test_build}"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release
  -DHYLE_COMPONENTS=error
  -DHYLE_BUILD_TESTS=ON
  -DHYLE_BUILD_BENCH=OFF
  -DHYLE_DEP_BUILD_DIR=${_error_dependency_cache}
  -DHYLE_DEPENDENCY_LOCK_FILE=${_fake_gtest_lock}
  -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES=${_fake_gtest_lock_attestation}
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=${_fake_gtest_source})
if(NOT IS_DIRECTORY "${_error_dependency_cache}/googletest-build")
  message(FATAL_ERROR
    "error-only tests ignored HYLE_DEP_BUILD_DIR='${_error_dependency_cache}'")
endif()
if(IS_DIRECTORY "${_error_test_build}/_deps")
  message(FATAL_ERROR
    "error-only tests leaked FetchContent state into '${_error_test_build}/_deps'")
endif()
file(STRINGS "${_error_test_build}/CMakeCache.txt" _gtest_cache_leaks
  REGEX "^(BUILD_GMOCK|INSTALL_GTEST|GTEST_HAS_ABSL|gtest_build_samples|gtest_build_tests):")
if(_gtest_cache_leaks)
  message(FATAL_ERROR
    "GoogleTest implementation options leaked into HYLE cache: ${_gtest_cache_leaks}")
endif()
file(STRINGS "${_error_test_build}/CMakeCache.txt" _instrumentation_default
  REGEX "^HYLE_TEST_INSTRUMENTATION:BOOL=")
if(NOT _instrumentation_default STREQUAL
   "HYLE_TEST_INSTRUMENTATION:BOOL=OFF")
  message(FATAL_ERROR
    "HYLE_TEST_INSTRUMENTATION default is not OFF: ${_instrumentation_default}")
endif()

# Shared factor targets must fail closed: Unix/Mach-O definitions are hidden
# unless a public declaration carries HYLE_<FACTOR>_API. Test the target
# property directly so a header-only selected product is not forced to invent a
# binary solely for this contract.
set(_shared_visibility_source
  "${_hyle_contract_root}/shared-visibility-source")
file(MAKE_DIRECTORY "${_shared_visibility_source}")
file(WRITE "${_shared_visibility_source}/CMakeLists.txt"
"cmake_minimum_required(VERSION 3.25)\n"
"project(hyle_shared_visibility_contract LANGUAGES CXX)\n"
"set(HYLE_COMPONENTS error)\n"
"set(HYLE_BUILD_TESTS OFF)\n"
"set(HYLE_BUILD_BENCH OFF)\n"
"set(HYLE_INSTALL OFF)\n"
"set(HYLE_STATIC OFF)\n"
"add_subdirectory(\"${HYLE_SOURCE_DIR}\" hyle)\n"
"foreach(_property IN ITEMS C_VISIBILITY_PRESET CXX_VISIBILITY_PRESET)\n"
"  get_target_property(_value hyle_error \${_property})\n"
"  if(NOT _value STREQUAL \"hidden\")\n"
"    message(FATAL_ERROR \"hyle_error \${_property}=\'\${_value}\'\")\n"
"  endif()\n"
"endforeach()\n"
"get_target_property(_hidden_inlines hyle_error VISIBILITY_INLINES_HIDDEN)\n"
"if(NOT _hidden_inlines)\n"
"  message(FATAL_ERROR \"hyle_error inline visibility is not hidden\")\n"
"endif()\n")
_hyle_contract_run("shared factor default-hidden visibility" TRUE
  "${CMAKE_COMMAND}" -S "${_shared_visibility_source}"
  -B "${_hyle_contract_root}/shared-visibility-build"
  ${_hyle_generator_args}
  -DCMAKE_BUILD_TYPE=Release)

message(STATUS "HYLE component contracts passed")
