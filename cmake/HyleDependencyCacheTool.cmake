cmake_minimum_required(VERSION 3.25)

# Script-mode maintenance entry for the shared locked-source cache. Example:
#   cmake -DHYLE_DEP_ACTION=sync -DHYLE_DEP_NAMES=highway\;xsimd \
#     -P cmake/HyleDependencyCacheTool.cmake

get_filename_component(HYLE_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(TO_CMAKE_PATH "${HYLE_SOURCE_DIR}" HYLE_SOURCE_DIR)
set(HYLE_BINARY_DIR "")
list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}")
include(HyleDependencyVersions)
include(HyleDependencyCache)
hyle_initialize_dependency_source_cache()

if(NOT DEFINED HYLE_DEP_ACTION OR "${HYLE_DEP_ACTION}" STREQUAL "")
  set(HYLE_DEP_ACTION status)
endif()
string(TOLOWER "${HYLE_DEP_ACTION}" _action)
if(NOT _action MATCHES "^(status|sync|check-updates|prune)$")
  message(FATAL_ERROR
    "HYLE_DEP_ACTION must be status, sync, check-updates, or prune; got "
    "'${HYLE_DEP_ACTION}'")
endif()

if(DEFINED HYLE_DEP_NAMES AND NOT "${HYLE_DEP_NAMES}" STREQUAL "")
  string(REPLACE "," ";" _names "${HYLE_DEP_NAMES}")
else()
  if(_action STREQUAL "check-updates")
    set(_names ${HYLE_DEPENDENCY_NAMES})
  else()
    set(_names
      ${HYLE_FETCHCONTENT_DEPENDENCIES}
      ${HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES})
  endif()
endif()
list(REMOVE_DUPLICATES _names)
foreach(_name IN LISTS _names)
  if(NOT _name IN_LIST HYLE_DEPENDENCY_NAMES)
    message(FATAL_ERROR
      "Unknown HYLE dependency '${_name}'. Known names: "
      "${HYLE_DEPENDENCY_NAMES}")
  endif()
endforeach()

function(_hyle_dependency_tool_repair dependency)
  if(NOT HYLE_DEP_REPAIR STREQUAL "YES")
    return()
  endif()
  _hyle_dependency_cache_prepare_root()
  hyle_dependency_cache_paths(
    "${dependency}" _archive _source _manifest _lock)
  file(LOCK "${_lock}" GUARD PROCESS TIMEOUT 0 RESULT_VARIABLE _lock_result)
  if(NOT _lock_result STREQUAL "0")
    message(FATAL_ERROR
      "Cannot repair '${dependency}'; its cache identity is busy: "
      "${_lock_result}")
  endif()
  string(TOUPPER "${dependency}" _upper)
  set(_prefix "HYLE_DEP_${_upper}")
  file(REMOVE_RECURSE
    "${_source}")
  file(REMOVE "${_manifest}")
  # Preserve a valid locked archive so source repair is offline-capable. A
  # corrupt archive is never reused and will be downloaded again by sync.
  if(EXISTS "${_archive}")
    file(SHA256 "${_archive}" _archive_sha256)
    if(NOT _archive_sha256 STREQUAL "${${_prefix}_ARCHIVE_SHA256}")
      file(REMOVE "${_archive}")
    endif()
  endif()
  # The identity-named staging path is computed below as part of normal prune;
  # remove it explicitly here as well.
  _hyle_dependency_cache_identity("${dependency}" _identity)
  file(REMOVE_RECURSE "${HYLE_DEP_SOURCE_CACHE}/staging/${_identity}")
  file(LOCK "${_lock}" RELEASE)
  message(STATUS "Removed selected cache identity for repair: ${dependency}")
endfunction()

function(_hyle_dependency_tool_prune_identity identity is_current)
  set(_lock "${HYLE_DEP_SOURCE_CACHE}/locks/${identity}.lock")
  file(LOCK "${_lock}" GUARD PROCESS TIMEOUT 0 RESULT_VARIABLE _lock_result)
  if(NOT _lock_result STREQUAL "0")
    message(WARNING
      "Skipping busy dependency cache identity '${identity}': ${_lock_result}")
    return()
  endif()
  # Interrupted staging is never a valid hit, including for a current lock
  # identity. Other artifacts are removed only when the identity is stale.
  file(REMOVE_RECURSE "${HYLE_DEP_SOURCE_CACHE}/staging/${identity}")
  if(NOT is_current)
    file(REMOVE_RECURSE "${HYLE_DEP_SOURCE_CACHE}/sources/${identity}")
    file(REMOVE
      "${HYLE_DEP_SOURCE_CACHE}/archives/${identity}.tar.gz"
      "${HYLE_DEP_SOURCE_CACHE}/manifests/${identity}.cmake")
    message(STATUS "Pruned stale HYLE dependency identity: ${identity}")
  endif()
  file(LOCK "${_lock}" RELEASE)
  if(NOT is_current)
    file(REMOVE "${_lock}")
  endif()
endfunction()

if(_action STREQUAL "status")
  _hyle_dependency_cache_prepare_root()
  message(STATUS "HYLE_DEP_SOURCE_CACHE=${HYLE_DEP_SOURCE_CACHE}")
  foreach(_name IN LISTS _names)
    if(_name IN_LIST HYLE_FETCHCONTENT_DEPENDENCIES OR
       _name IN_LIST HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES)
      hyle_dependency_cache_paths(
        "${_name}" _archive _source _manifest _lock)
      if(IS_DIRECTORY "${_source}")
        hyle_verify_locked_expanded_source("${_name}" "${_source}")
        message(STATUS "${_name}: CURRENT @ ${_source}")
      elseif(EXISTS "${_source}")
        message(FATAL_ERROR
          "${_name}: INVALID non-directory cache entry @ ${_source}")
      else()
        message(STATUS "${_name}: MISSING")
      endif()
    else()
      string(TOUPPER "${_name}" _upper)
      set(_prefix "HYLE_DEP_${_upper}")
      message(STATUS
        "${_name}: ${${_prefix}_CLASS} (not source-cached)")
    endif()
  endforeach()
elseif(_action STREQUAL "sync")
  foreach(_name IN LISTS _names)
    if(NOT _name IN_LIST HYLE_FETCHCONTENT_DEPENDENCIES AND
       NOT _name IN_LIST HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES)
      message(FATAL_ERROR
        "Cannot sync '${_name}': class is not fetchcontent/external_checkout")
    endif()
    _hyle_dependency_tool_repair("${_name}")
    hyle_dependency_cache_resolve("${_name}" _source)
    message(STATUS "${_name}: READY @ ${_source}")
  endforeach()
elseif(_action STREQUAL "check-updates")
  find_package(Git REQUIRED)
  set(_failures "")
  foreach(_name IN LISTS _names)
    string(TOUPPER "${_name}" _upper)
    set(_prefix "HYLE_DEP_${_upper}")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" ls-remote
              "${${_prefix}_REPOSITORY}" HEAD
      RESULT_VARIABLE _result
      OUTPUT_VARIABLE _stdout
      ERROR_VARIABLE _stderr
      OUTPUT_STRIP_TRAILING_WHITESPACE
      ERROR_STRIP_TRAILING_WHITESPACE
      TIMEOUT 45)
    string(REGEX MATCH "^[0-9a-f]+" _head "${_stdout}")
    string(LENGTH "${_head}" _head_length)
    if(NOT _result EQUAL 0 OR NOT _head_length EQUAL 40)
      list(APPEND _failures "${_name}")
      message(WARNING
        "${_name}: cannot query upstream HEAD (${_result}): ${_stderr}")
      continue()
    endif()
    if(_head STREQUAL "${${_prefix}_COMMIT}")
      message(STATUS
        "${_name}: LOCK_EQUALS_UPSTREAM_HEAD ${_head}")
    elseif("${${_prefix}_VERSION}" MATCHES "^snapshot-")
      message(STATUS
        "${_name}: SNAPSHOT_UPDATE_CANDIDATE\n"
        "  locked: ${${_prefix}_COMMIT}\n  HEAD:   ${_head}")
    else()
      message(STATUS
        "${_name}: PINNED_RELEASE_OR_TRAIN\n"
        "  version: ${${_prefix}_VERSION}\n"
        "  locked:  ${${_prefix}_COMMIT}\n"
        "  HEAD:    ${_head}\n"
        "  HEAD difference is informational, not a release-upgrade verdict.")
    endif()
  endforeach()
  if(_failures)
    message(FATAL_ERROR
      "Could not check upstream state for: ${_failures}")
  endif()
elseif(_action STREQUAL "prune")
  if(NOT HYLE_DEP_PRUNE_CONFIRM STREQUAL "YES")
    message(FATAL_ERROR
      "Prune is destructive. Re-run with -DHYLE_DEP_PRUNE_CONFIRM=YES.")
  endif()
  _hyle_dependency_cache_prepare_root()
  set(_current_identities "")
  foreach(_name IN LISTS HYLE_FETCHCONTENT_DEPENDENCIES
                         HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES)
    _hyle_dependency_cache_identity("${_name}" _identity)
    list(APPEND _current_identities "${_identity}")
  endforeach()
  set(_found_identities "")
  file(GLOB _source_entries LIST_DIRECTORIES TRUE
       "${HYLE_DEP_SOURCE_CACHE}/sources/*")
  file(GLOB _staging_entries LIST_DIRECTORIES TRUE
       "${HYLE_DEP_SOURCE_CACHE}/staging/*")
  file(GLOB _archive_entries LIST_DIRECTORIES FALSE
       "${HYLE_DEP_SOURCE_CACHE}/archives/*.tar.gz")
  file(GLOB _manifest_entries LIST_DIRECTORIES FALSE
       "${HYLE_DEP_SOURCE_CACHE}/manifests/*.cmake")
  file(GLOB _lock_entries LIST_DIRECTORIES FALSE
       "${HYLE_DEP_SOURCE_CACHE}/locks/*.lock")
  foreach(_entry IN LISTS _source_entries _staging_entries)
    get_filename_component(_identity "${_entry}" NAME)
    list(APPEND _found_identities "${_identity}")
  endforeach()
  foreach(_entry IN LISTS _archive_entries)
    get_filename_component(_identity "${_entry}" NAME)
    string(REGEX REPLACE "\\.tar\\.gz$" "" _identity "${_identity}")
    list(APPEND _found_identities "${_identity}")
  endforeach()
  foreach(_entry IN LISTS _manifest_entries)
    get_filename_component(_identity "${_entry}" NAME_WE)
    list(APPEND _found_identities "${_identity}")
  endforeach()
  foreach(_entry IN LISTS _lock_entries)
    get_filename_component(_identity "${_entry}" NAME_WE)
    list(APPEND _found_identities "${_identity}")
  endforeach()
  list(REMOVE_DUPLICATES _found_identities)
  foreach(_identity IN LISTS _found_identities)
    if(NOT _identity MATCHES "^[a-z0-9_.-]+$")
      message(WARNING
        "Leaving unrecognized cache entry untouched: ${_identity}")
      continue()
    endif()
    if(_identity IN_LIST _current_identities)
      _hyle_dependency_tool_prune_identity("${_identity}" TRUE)
    else()
      _hyle_dependency_tool_prune_identity("${_identity}" FALSE)
    endif()
  endforeach()
endif()
