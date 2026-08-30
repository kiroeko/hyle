include_guard(GLOBAL)

# Platform-independent dependency sources are immutable resources shared by
# build trees.  FetchContent binary/subbuild state remains build-local; this
# file only owns the locked archive/source cache.

set(_HYLE_DEPENDENCY_CACHE_MARKER_TEXT
    "HYLE dependency source cache\nschema=1\n")

function(_hyle_dependency_cache_normalize_absolute input base output)
  set(_path "${input}")
  file(TO_CMAKE_PATH "${_path}" _path)
  if(NOT IS_ABSOLUTE "${_path}")
    cmake_path(ABSOLUTE_PATH _path BASE_DIRECTORY "${base}" NORMALIZE
               OUTPUT_VARIABLE _path)
  else()
    cmake_path(NORMAL_PATH _path OUTPUT_VARIABLE _path)
  endif()
  set(${output} "${_path}" PARENT_SCOPE)
endfunction()

function(_hyle_dependency_cache_path_is_within child parent output)
  file(RELATIVE_PATH _relative "${parent}" "${child}")
  file(TO_CMAKE_PATH "${_relative}" _relative)
  if(_relative STREQUAL "" OR
     (NOT IS_ABSOLUTE "${_relative}" AND
      NOT _relative STREQUAL ".." AND
      NOT _relative MATCHES "^\\.\\./"))
    set(${output} TRUE PARENT_SCOPE)
  else()
    set(${output} FALSE PARENT_SCOPE)
  endif()
endfunction()

function(_hyle_dependency_cache_default_root output)
  if(DEFINED ENV{HYLE_DEP_SOURCE_CACHE} AND
     NOT "$ENV{HYLE_DEP_SOURCE_CACHE}" STREQUAL "")
    set(_root "$ENV{HYLE_DEP_SOURCE_CACHE}")
  elseif(WIN32 AND DEFINED ENV{LOCALAPPDATA} AND
         NOT "$ENV{LOCALAPPDATA}" STREQUAL "")
    set(_root "$ENV{LOCALAPPDATA}/hyle/dependencies/v1")
  elseif(APPLE AND DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    set(_root "$ENV{HOME}/Library/Caches/hyle/dependencies/v1")
  elseif(DEFINED ENV{XDG_CACHE_HOME} AND
         NOT "$ENV{XDG_CACHE_HOME}" STREQUAL "")
    set(_root "$ENV{XDG_CACHE_HOME}/hyle/dependencies/v1")
  elseif(DEFINED ENV{HOME} AND NOT "$ENV{HOME}" STREQUAL "")
    set(_root "$ENV{HOME}/.cache/hyle/dependencies/v1")
  else()
    message(FATAL_ERROR
      "Cannot choose a shared HYLE dependency source cache. Set the absolute "
      "HYLE_DEP_SOURCE_CACHE CMake variable or environment variable.")
  endif()
  set(${output} "${_root}" PARENT_SCOPE)
endfunction()

function(hyle_initialize_dependency_source_cache)
  if(DEFINED HYLE_DEP_SOURCE_CACHE AND
     NOT "${HYLE_DEP_SOURCE_CACHE}" STREQUAL "")
    set(_input "${HYLE_DEP_SOURCE_CACHE}")
  else()
    _hyle_dependency_cache_default_root(_input)
  endif()
  _hyle_dependency_cache_normalize_absolute(
    "${_input}" "${HYLE_DEPENDENCY_SOURCE_ROOT}" _normalized)

  _hyle_dependency_cache_path_is_within(
    "${_normalized}" "${HYLE_DEPENDENCY_SOURCE_ROOT}" _inside_source)
  if(_inside_source)
    message(FATAL_ERROR
      "HYLE_DEP_SOURCE_CACHE must live outside the HYLE source checkout.\n"
      "  source: ${HYLE_DEPENDENCY_SOURCE_ROOT}\n"
      "  cache:  ${_normalized}")
  endif()
  if(DEFINED HYLE_BINARY_DIR AND NOT "${HYLE_BINARY_DIR}" STREQUAL "")
    _hyle_dependency_cache_normalize_absolute(
      "${HYLE_BINARY_DIR}" "${HYLE_DEPENDENCY_SOURCE_ROOT}"
      _normalized_binary)
    _hyle_dependency_cache_path_is_within(
      "${_normalized}" "${_normalized_binary}" _inside_binary)
    if(_inside_binary)
      message(FATAL_ERROR
        "HYLE_DEP_SOURCE_CACHE must not live inside the current build tree.\n"
        "  build: ${_normalized_binary}\n"
        "  cache: ${_normalized}")
    endif()
  endif()

  set(HYLE_DEP_SOURCE_CACHE "${_normalized}" CACHE PATH
      "Shared immutable source cache for HYLE locked dependencies" FORCE)
  set(HYLE_DEP_SOURCE_CACHE "${_normalized}" PARENT_SCOPE)
  mark_as_advanced(HYLE_DEP_SOURCE_CACHE)
endfunction()

function(_hyle_dependency_cache_prepare_root)
  if(NOT DEFINED HYLE_DEP_SOURCE_CACHE OR
     "${HYLE_DEP_SOURCE_CACHE}" STREQUAL "")
    message(FATAL_ERROR
      "hyle_initialize_dependency_source_cache() must run first")
  endif()
  # Directory creation is the synchronization-safe first filesystem action.
  # Checking EXISTS and IS_DIRECTORY separately before this point creates a
  # Windows TOCTOU window when two first-time configures initialize the same
  # absent root. A foreign file still makes MAKE_DIRECTORY fail; a symlink is
  # rejected immediately afterwards before HYLE writes cache content.
  file(MAKE_DIRECTORY "${HYLE_DEP_SOURCE_CACHE}")
  if(NOT IS_DIRECTORY "${HYLE_DEP_SOURCE_CACHE}" OR
     IS_SYMLINK "${HYLE_DEP_SOURCE_CACHE}")
    message(FATAL_ERROR
      "HYLE_DEP_SOURCE_CACHE must be a real directory: "
      "${HYLE_DEP_SOURCE_CACHE}")
  endif()
  set(_marker "${HYLE_DEP_SOURCE_CACHE}/.hyle-dependency-cache-v1")
  set(_initialization_lock
      "${HYLE_DEP_SOURCE_CACHE}/.hyle-dependency-cache-init.lock")
  # Refuse foreign data before creating the reserved initialization lock. The
  # lock itself is the only entry permitted in the small race window between
  # another process creating the root and publishing its marker.
  if(EXISTS "${HYLE_DEP_SOURCE_CACHE}" AND NOT EXISTS "${_marker}")
    file(GLOB _unmarked_entries LIST_DIRECTORIES TRUE
         "${HYLE_DEP_SOURCE_CACHE}/*")
    list(REMOVE_ITEM _unmarked_entries "${_initialization_lock}")
    if(_unmarked_entries)
      message(FATAL_ERROR
        "Refusing non-empty unmarked HYLE_DEP_SOURCE_CACHE:\n"
        "  ${HYLE_DEP_SOURCE_CACHE}\n"
        "Choose an empty directory or a cache created by HYLE.")
    endif()
  endif()
  file(LOCK "${_initialization_lock}" GUARD FUNCTION TIMEOUT 60
       RESULT_VARIABLE _initialization_lock_result)
  if(NOT _initialization_lock_result STREQUAL "0")
    message(FATAL_ERROR
      "Timed out initializing HYLE dependency source cache '${HYLE_DEP_SOURCE_CACHE}': "
      "${_initialization_lock_result}")
  endif()
  # Repeat the foreign-content check under the lock: the root may have been
  # absent during the optimistic check above.
  if(NOT EXISTS "${_marker}")
    file(GLOB _unmarked_entries LIST_DIRECTORIES TRUE
         "${HYLE_DEP_SOURCE_CACHE}/*")
    list(REMOVE_ITEM _unmarked_entries "${_initialization_lock}")
    if(_unmarked_entries)
      message(FATAL_ERROR
        "Refusing non-empty unmarked HYLE_DEP_SOURCE_CACHE:\n"
        "  ${HYLE_DEP_SOURCE_CACHE}\n"
        "Choose an empty directory or a cache created by HYLE.")
    endif()
  endif()
  if(EXISTS "${_marker}")
    file(READ "${_marker}" _marker_content)
    if(NOT _marker_content STREQUAL _HYLE_DEPENDENCY_CACHE_MARKER_TEXT)
      message(FATAL_ERROR
        "Unsupported or corrupt HYLE dependency cache marker: ${_marker}")
    endif()
  else()
    file(WRITE "${_marker}" "${_HYLE_DEPENDENCY_CACHE_MARKER_TEXT}")
  endif()
  file(MAKE_DIRECTORY
    "${HYLE_DEP_SOURCE_CACHE}/archives"
    "${HYLE_DEP_SOURCE_CACHE}/locks"
    "${HYLE_DEP_SOURCE_CACHE}/manifests"
    "${HYLE_DEP_SOURCE_CACHE}/sources"
    "${HYLE_DEP_SOURCE_CACHE}/staging")
endfunction()

function(_hyle_dependency_cache_identity dependency output)
  if(NOT dependency IN_LIST HYLE_FETCHCONTENT_DEPENDENCIES AND
     NOT dependency IN_LIST HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES)
    message(FATAL_ERROR
      "Dependency '${dependency}' has no external source-cache identity")
  endif()
  string(TOUPPER "${dependency}" _upper)
  set(_prefix "HYLE_DEP_${_upper}")
  string(SUBSTRING "${${_prefix}_COMMIT}" 0 12 _revision_short)
  string(SUBSTRING "${${_prefix}_EXPANDED_TREE_SHA256}" 0 12 _tree_short)
  set(${output}
      "${dependency}-${_revision_short}-${_tree_short}" PARENT_SCOPE)
endfunction()

function(hyle_dependency_cache_paths dependency archive_out source_out
         manifest_out lock_out)
  _hyle_dependency_cache_identity("${dependency}" _identity)
  set(${archive_out}
      "${HYLE_DEP_SOURCE_CACHE}/archives/${_identity}.tar.gz" PARENT_SCOPE)
  set(${source_out}
      "${HYLE_DEP_SOURCE_CACHE}/sources/${_identity}" PARENT_SCOPE)
  set(${manifest_out}
      "${HYLE_DEP_SOURCE_CACHE}/manifests/${_identity}.cmake" PARENT_SCOPE)
  set(${lock_out}
      "${HYLE_DEP_SOURCE_CACHE}/locks/${_identity}.lock" PARENT_SCOPE)
endfunction()

function(_hyle_dependency_cache_write_manifest dependency manifest)
  string(TOUPPER "${dependency}" _upper)
  set(_prefix "HYLE_DEP_${_upper}")
  set(_temporary "${manifest}.tmp")
  file(WRITE "${_temporary}"
    "set(HYLE_CACHE_DEPENDENCY [==[${dependency}]==])\n"
    "set(HYLE_CACHE_REVISION [==[${${_prefix}_COMMIT}]==])\n"
    "set(HYLE_CACHE_ARCHIVE_SHA256 [==[${${_prefix}_ARCHIVE_SHA256}]==])\n"
    "set(HYLE_CACHE_EXPANDED_TREE_SHA256 [==[${${_prefix}_EXPANDED_TREE_SHA256}]==])\n")
  file(RENAME "${_temporary}" "${manifest}" RESULT _rename_result)
  if(NOT _rename_result STREQUAL "0")
    message(FATAL_ERROR
      "Cannot publish HYLE dependency cache manifest '${manifest}': "
      "${_rename_result}")
  endif()
endfunction()

function(_hyle_dependency_cache_populate dependency)
  _hyle_dependency_cache_prepare_root()
  hyle_dependency_cache_paths(
    "${dependency}" _archive _source _manifest _lock)
  file(LOCK "${_lock}" GUARD FUNCTION TIMEOUT 600 RESULT_VARIABLE _lock_result)
  if(NOT _lock_result STREQUAL "0")
    message(FATAL_ERROR
      "Timed out waiting for dependency cache lock '${_lock}': "
      "${_lock_result}")
  endif()

  # Another configure may have completed while this process waited.
  if(IS_DIRECTORY "${_source}")
    hyle_verify_locked_expanded_source("${dependency}" "${_source}")
    return()
  elseif(EXISTS "${_source}")
    message(FATAL_ERROR
      "Locked dependency cache source is not a directory: ${_source}")
  endif()

  string(TOUPPER "${dependency}" _upper)
  set(_prefix "HYLE_DEP_${_upper}")
  if(EXISTS "${_archive}")
    file(SHA256 "${_archive}" _archive_sha256)
    if(NOT _archive_sha256 STREQUAL "${${_prefix}_ARCHIVE_SHA256}")
      message(FATAL_ERROR
        "Cached archive for '${dependency}' is corrupt.\n"
        "  path:     ${_archive}\n"
        "  expected: ${${_prefix}_ARCHIVE_SHA256}\n"
        "  actual:   ${_archive_sha256}\n"
        "Remove this cache identity with the guarded prune/repair workflow "
        "before retrying; configure will not silently trust or overwrite it.")
    endif()
  else()
    set(_archive_partial "${_archive}.partial")
    file(REMOVE "${_archive_partial}")
    message(STATUS
      "HYLE dependency cache miss: downloading ${dependency} "
      "${${_prefix}_VERSION}")
    file(DOWNLOAD
      "${${_prefix}_ARCHIVE_URL}" "${_archive_partial}"
      EXPECTED_HASH "SHA256=${${_prefix}_ARCHIVE_SHA256}"
      TLS_VERIFY ON
      INACTIVITY_TIMEOUT 60
      TIMEOUT 600
      STATUS _download_status
      LOG _download_log
      SHOW_PROGRESS)
    list(GET _download_status 0 _download_code)
    list(GET _download_status 1 _download_message)
    if(NOT _download_code EQUAL 0)
      file(REMOVE "${_archive_partial}")
      message(FATAL_ERROR
        "Cannot download locked dependency '${dependency}' from "
        "${${_prefix}_ARCHIVE_URL}: ${_download_message}")
    endif()
    file(RENAME "${_archive_partial}" "${_archive}"
         RESULT _archive_rename_result)
    if(NOT _archive_rename_result STREQUAL "0")
      message(FATAL_ERROR
        "Cannot publish locked archive '${_archive}': "
        "${_archive_rename_result}")
    endif()
  endif()

  _hyle_dependency_cache_identity("${dependency}" _identity)
  set(_staging "${HYLE_DEP_SOURCE_CACHE}/staging/${_identity}")
  # A prior interrupted writer never becomes a hit. The per-identity lock makes
  # removing its private staging directory safe.
  file(REMOVE_RECURSE "${_staging}")
  file(MAKE_DIRECTORY "${_staging}/extract")
  file(ARCHIVE_EXTRACT INPUT "${_archive}"
       DESTINATION "${_staging}/extract")
  file(GLOB _archive_roots LIST_DIRECTORIES TRUE
       "${_staging}/extract/*")
  list(LENGTH _archive_roots _archive_root_count)
  if(NOT _archive_root_count EQUAL 1)
    message(FATAL_ERROR
      "Locked archive for '${dependency}' must contain exactly one top-level "
      "directory; found ${_archive_root_count}")
  endif()
  list(GET _archive_roots 0 _expanded_root)
  if(NOT IS_DIRECTORY "${_expanded_root}" OR IS_SYMLINK "${_expanded_root}")
    message(FATAL_ERROR
      "Locked archive for '${dependency}' has an invalid top-level entry: "
      "${_expanded_root}")
  endif()
  hyle_verify_locked_expanded_source("${dependency}" "${_expanded_root}")

  file(RENAME "${_expanded_root}" "${_source}" RESULT _source_rename_result)
  if(NOT _source_rename_result STREQUAL "0")
    message(FATAL_ERROR
      "Cannot publish locked source '${_source}': ${_source_rename_result}")
  endif()
  _hyle_dependency_cache_write_manifest("${dependency}" "${_manifest}")
  file(REMOVE_RECURSE "${_staging}")
  message(STATUS
    "HYLE dependency source cached: ${dependency} @ ${_source}")
endfunction()

function(hyle_dependency_cache_resolve dependency output)
  _hyle_dependency_cache_prepare_root()
  hyle_dependency_cache_paths(
    "${dependency}" _archive _source _manifest _lock)
  if(IS_DIRECTORY "${_source}")
    hyle_verify_locked_expanded_source("${dependency}" "${_source}")
  elseif(EXISTS "${_source}")
    message(FATAL_ERROR
      "Locked dependency cache source is not a directory: ${_source}")
  else()
    # Disconnected means no network, not "never reconstruct local state". A
    # verified locked archive is sufficient to rebuild a missing/corrupt
    # expanded tree; _hyle_dependency_cache_populate() rechecks its SHA before
    # extraction. Only a cache that lacks both source and archive is a true
    # disconnected miss.
    if(FETCHCONTENT_FULLY_DISCONNECTED AND NOT EXISTS "${_archive}")
      message(FATAL_ERROR
        "Locked dependency '${dependency}' is missing from the shared source "
        "cache while FETCHCONTENT_FULLY_DISCONNECTED=ON.\n"
        "  cache: ${HYLE_DEP_SOURCE_CACHE}\n"
        "Prepare it on a connected machine with:\n"
        "  cmake -DHYLE_DEP_SOURCE_CACHE=${HYLE_DEP_SOURCE_CACHE} "
        "-DHYLE_DEP_ACTION=sync -DHYLE_DEP_NAMES=${dependency} "
        "-P cmake/HyleDependencyCacheTool.cmake")
    endif()
    _hyle_dependency_cache_populate("${dependency}")
    hyle_verify_locked_expanded_source("${dependency}" "${_source}")
  endif()
  set(${output} "${_source}" PARENT_SCOPE)
endfunction()
