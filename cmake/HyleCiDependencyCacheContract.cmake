# Lifecycle contract for the shared immutable dependency-source cache. The
# caller supplies one already populated real lock identity; this script copies
# only its verified archive into isolated scratch and never mutates the seed.

cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS
    HYLE_SOURCE_DIR HYLE_SMOKE_ROOT HYLE_SEED_SOURCE_CACHE)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=<absolute-path>")
  endif()
  get_filename_component(${_required} "${${_required}}" ABSOLUTE)
  file(TO_CMAKE_PATH "${${_required}}" ${_required})
endforeach()

set(HYLE_DEP_SOURCE_CACHE "${HYLE_SEED_SOURCE_CACHE}")
set(HYLE_BINARY_DIR "")
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyVersions.cmake")
include("${HYLE_SOURCE_DIR}/cmake/HyleDependencyCache.cmake")
hyle_initialize_dependency_source_cache()
hyle_dependency_cache_resolve(highway _seed_source)
hyle_dependency_cache_paths(
  highway _seed_archive _seed_locked_source _seed_manifest _seed_lock)
if(NOT EXISTS "${_seed_archive}" OR
   NOT IS_DIRECTORY "${_seed_locked_source}")
  message(FATAL_ERROR
    "The dependency-cache contract seed is incomplete: "
    "${HYLE_SEED_SOURCE_CACHE}")
endif()

set(_tool "${HYLE_SOURCE_DIR}/cmake/HyleDependencyCacheTool.cmake")
set(_root "${HYLE_SMOKE_ROOT}/dependency-source-cache-contract")
set(_cache "${_root}/cache")
set(_missing_cache "${_root}/missing-cache")
set(_unmarked_cache "${_root}/unmarked-cache")
file(REMOVE_RECURSE "${_root}")
file(MAKE_DIRECTORY "${_root}")

function(_hyle_cache_contract_run label expect_success)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr)
  set(_output "${_stdout}\n${_stderr}")
  set(HYLE_CACHE_CONTRACT_LAST_OUTPUT "${_output}" PARENT_SCOPE)
  if(expect_success AND NOT _result EQUAL 0)
    message(FATAL_ERROR
      "${label} failed with exit code ${_result}:\n${_output}")
  elseif(NOT expect_success AND _result EQUAL 0)
    message(FATAL_ERROR "${label} unexpectedly succeeded:\n${_output}")
  endif()
endfunction()

# Initialize a marked empty cache concurrently, then seed only the immutable
# archive. The two concurrent sync processes must serialize
# extraction/publication and both finish on one complete identity. A stale
# staging directory simulates an interrupted former writer and must never
# become a hit.
execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -DHYLE_DEP_SOURCE_CACHE=${_cache}
    -DHYLE_DEP_ACTION=status
    -DHYLE_DEP_NAMES=highway
    -P "${_tool}"
  COMMAND "${CMAKE_COMMAND}"
    -DHYLE_DEP_SOURCE_CACHE=${_cache}
    -DHYLE_DEP_ACTION=status
    -DHYLE_DEP_NAMES=highway
    -P "${_tool}"
  RESULTS_VARIABLE _initialization_results
  OUTPUT_VARIABLE _initialization_stdout
  ERROR_VARIABLE _initialization_stderr)
if(NOT _initialization_results STREQUAL "0;0")
  message(FATAL_ERROR
    "Concurrent cache initialization failed (${_initialization_results}):\n"
    "${_initialization_stdout}\n${_initialization_stderr}")
endif()
get_filename_component(_archive_name "${_seed_archive}" NAME)
file(COPY_FILE "${_seed_archive}"
  "${_cache}/archives/${_archive_name}" ONLY_IF_DIFFERENT)
_hyle_dependency_cache_identity(highway _identity)
file(MAKE_DIRECTORY "${_cache}/staging/${_identity}/interrupted")
file(WRITE
  "${_cache}/staging/${_identity}/interrupted/partial.txt" "partial\n")

execute_process(
  COMMAND "${CMAKE_COMMAND}"
    -DHYLE_DEP_SOURCE_CACHE=${_cache}
    -DHYLE_DEP_ACTION=sync
    -DHYLE_DEP_NAMES=highway
    -P "${_tool}"
  COMMAND "${CMAKE_COMMAND}"
    -DHYLE_DEP_SOURCE_CACHE=${_cache}
    -DHYLE_DEP_ACTION=sync
    -DHYLE_DEP_NAMES=highway
    -P "${_tool}"
  RESULTS_VARIABLE _concurrent_results
  OUTPUT_VARIABLE _concurrent_stdout
  ERROR_VARIABLE _concurrent_stderr)
if(NOT _concurrent_results STREQUAL "0;0")
  message(FATAL_ERROR
    "Concurrent cache population failed (${_concurrent_results}):\n"
    "${_concurrent_stdout}\n${_concurrent_stderr}")
endif()
file(GLOB _published_sources LIST_DIRECTORIES TRUE "${_cache}/sources/*")
list(LENGTH _published_sources _published_source_count)
if(NOT _published_source_count EQUAL 1 OR
   NOT IS_DIRECTORY "${_cache}/sources/${_identity}" OR
   EXISTS "${_cache}/staging/${_identity}")
  message(FATAL_ERROR
    "Concurrent population did not publish exactly one complete identity")
endif()

# A disconnected hit revalidates but must not rewrite either source or archive.
set(_archive "${_cache}/archives/${_archive_name}")
set(_readme "${_cache}/sources/${_identity}/README.md")
file(TIMESTAMP "${_archive}" _archive_before "%s" UTC)
file(TIMESTAMP "${_readme}" _source_before "%s" UTC)
file(SHA256 "${_archive}" _archive_hash_before)
file(SHA256 "${_readme}" _source_hash_before)
_hyle_cache_contract_run("disconnected shared-cache hit" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_DEP_SOURCE_CACHE=${_cache}
  -DHYLE_DEP_ACTION=sync
  -DHYLE_DEP_NAMES=highway
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  -P "${_tool}")
file(TIMESTAMP "${_archive}" _archive_after "%s" UTC)
file(TIMESTAMP "${_readme}" _source_after "%s" UTC)
file(SHA256 "${_archive}" _archive_hash_after)
file(SHA256 "${_readme}" _source_hash_after)
if(NOT _archive_before STREQUAL _archive_after OR
   NOT _source_before STREQUAL _source_after OR
   NOT _archive_hash_before STREQUAL _archive_hash_after OR
   NOT _source_hash_before STREQUAL _source_hash_after)
  message(FATAL_ERROR "A shared-cache hit rewrote immutable cache content")
endif()

# Source corruption fails closed. Explicit named repair preserves the valid
# archive and therefore succeeds even under the disconnected gate.
file(APPEND "${_readme}" "\ncorruption fixture\n")
_hyle_cache_contract_run("corrupt source rejection" FALSE
  "${CMAKE_COMMAND}"
  -DHYLE_DEP_SOURCE_CACHE=${_cache}
  -DHYLE_DEP_ACTION=status
  -DHYLE_DEP_NAMES=highway
  -P "${_tool}")
if(NOT HYLE_CACHE_CONTRACT_LAST_OUTPUT MATCHES
   "expanded source tree mismatch")
  message(FATAL_ERROR
    "Corrupt source failed for the wrong reason:\n"
    "${HYLE_CACHE_CONTRACT_LAST_OUTPUT}")
endif()
_hyle_cache_contract_run("offline named source repair" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_DEP_SOURCE_CACHE=${_cache}
  -DHYLE_DEP_ACTION=sync
  -DHYLE_DEP_NAMES=highway
  -DHYLE_DEP_REPAIR=YES
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  -P "${_tool}")
file(TIMESTAMP "${_archive}" _archive_after_repair "%s" UTC)
file(SHA256 "${_archive}" _archive_hash_after_repair)
if(NOT _archive_before STREQUAL _archive_after_repair OR
   NOT _archive_hash_before STREQUAL _archive_hash_after_repair)
  message(FATAL_ERROR "Source repair rewrote the valid locked archive")
endif()

# Missing disconnected input must fail without network access.
_hyle_cache_contract_run("initialize disconnected-miss cache" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_DEP_SOURCE_CACHE=${_missing_cache}
  -DHYLE_DEP_ACTION=status
  -DHYLE_DEP_NAMES=highway
  -P "${_tool}")
_hyle_cache_contract_run("disconnected cache miss" FALSE
  "${CMAKE_COMMAND}"
  -DHYLE_DEP_SOURCE_CACHE=${_missing_cache}
  -DHYLE_DEP_ACTION=sync
  -DHYLE_DEP_NAMES=highway
  -DFETCHCONTENT_FULLY_DISCONNECTED=ON
  -P "${_tool}")
if(NOT HYLE_CACHE_CONTRACT_LAST_OUTPUT MATCHES
   "missing from the shared source cache")
  message(FATAL_ERROR
    "Disconnected miss failed for the wrong reason:\n"
    "${HYLE_CACHE_CONTRACT_LAST_OUTPUT}")
endif()

# Prune is restricted to marked caches, requires confirmation, and deletes
# only identities not referenced by the current central lock.
set(_stale_identity "stale-000000000000-000000000000")
file(MAKE_DIRECTORY "${_cache}/sources/${_stale_identity}")
file(WRITE "${_cache}/sources/${_stale_identity}/stale.txt" "stale\n")
file(WRITE "${_cache}/archives/${_stale_identity}.tar.gz" "stale\n")
file(WRITE "${_cache}/manifests/${_stale_identity}.cmake" "# stale\n")
_hyle_cache_contract_run("unconfirmed cache prune" FALSE
  "${CMAKE_COMMAND}"
  -DHYLE_DEP_SOURCE_CACHE=${_cache}
  -DHYLE_DEP_ACTION=prune
  -P "${_tool}")
if(NOT EXISTS "${_cache}/sources/${_stale_identity}")
  message(FATAL_ERROR "Unconfirmed prune changed the source cache")
endif()
_hyle_cache_contract_run("confirmed stale cache prune" TRUE
  "${CMAKE_COMMAND}"
  -DHYLE_DEP_SOURCE_CACHE=${_cache}
  -DHYLE_DEP_ACTION=prune
  -DHYLE_DEP_PRUNE_CONFIRM=YES
  -P "${_tool}")
if(EXISTS "${_cache}/sources/${_stale_identity}" OR
   EXISTS "${_cache}/archives/${_stale_identity}.tar.gz" OR
   NOT IS_DIRECTORY "${_cache}/sources/${_identity}")
  message(FATAL_ERROR "Prune did not preserve only the current identity")
endif()

file(MAKE_DIRECTORY "${_unmarked_cache}")
file(WRITE "${_unmarked_cache}/keep.txt" "not a HYLE cache\n")
_hyle_cache_contract_run("unmarked cache refusal" FALSE
  "${CMAKE_COMMAND}"
  -DHYLE_DEP_SOURCE_CACHE=${_unmarked_cache}
  -DHYLE_DEP_ACTION=status
  -DHYLE_DEP_NAMES=highway
  -P "${_tool}")
if(NOT EXISTS "${_unmarked_cache}/keep.txt")
  message(FATAL_ERROR "Unmarked cache refusal changed foreign content")
endif()

message(STATUS "HYLE shared dependency source-cache lifecycle passed")
