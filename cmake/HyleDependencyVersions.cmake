include_guard(GLOBAL)

# HyleDependencyLock.json is the only hand-maintained dependency ledger.  This
# module validates it, projects its records to the legacy HYLE_DEP_* variables,
# and proves that each in-tree snapshot still matches the locked content.
get_filename_component(_hyle_default_dependency_lock_file
  "${CMAKE_CURRENT_LIST_DIR}/HyleDependencyLock.json" ABSOLUTE)
if(NOT DEFINED HYLE_DEPENDENCY_LOCK_FILE)
  set(HYLE_DEPENDENCY_LOCK_FILE "${_hyle_default_dependency_lock_file}")
endif()
get_filename_component(HYLE_DEPENDENCY_LOCK_FILE
                       "${HYLE_DEPENDENCY_LOCK_FILE}" ABSOLUTE)
if(NOT HYLE_DEPENDENCY_LOCK_FILE STREQUAL
       _hyle_default_dependency_lock_file)
  # A normal -D must not be able to replace the ledger that defines every
  # downstream URL, hash and source identity. Contract fixtures that need a
  # deliberately malformed lock execute trusted CMake code first and attest
  # the one exact alternate path through this non-cache GLOBAL property.
  get_property(_hyle_attested_dependency_lock_file GLOBAL PROPERTY
    HYLE_INTERNAL_ATTESTED_DEPENDENCY_LOCK_FILE)
  if(NOT _hyle_attested_dependency_lock_file STREQUAL
         HYLE_DEPENDENCY_LOCK_FILE)
    message(FATAL_ERROR
      "Refusing untrusted HYLE_DEPENDENCY_LOCK_FILE override "
      "'${HYLE_DEPENDENCY_LOCK_FILE}'. The checked-in central lock is "
      "'${_hyle_default_dependency_lock_file}'.")
  endif()
endif()
get_filename_component(_hyle_dependency_source_root
                       "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(HYLE_DEPENDENCY_SOURCE_ROOT "${_hyle_dependency_source_root}")

if(NOT EXISTS "${HYLE_DEPENDENCY_LOCK_FILE}")
  message(FATAL_ERROR
    "HYLE dependency lock does not exist: ${HYLE_DEPENDENCY_LOCK_FILE}")
endif()
file(READ "${HYLE_DEPENDENCY_LOCK_FILE}" _hyle_dependency_lock_json)

function(_hyle_dependency_json_get output)
  string(JSON _value ERROR_VARIABLE _error
         GET "${_hyle_dependency_lock_json}" ${ARGN})
  if(NOT _error STREQUAL "NOTFOUND")
    string(JOIN "." _path ${ARGN})
    message(FATAL_ERROR
      "Invalid HYLE dependency lock at '${_path}': ${_error}")
  endif()
  set(${output} "${_value}" PARENT_SCOPE)
endfunction()

function(_hyle_dependency_require_hex label value length)
  string(LENGTH "${value}" _actual_length)
  if(NOT _actual_length EQUAL ${length} OR
     NOT "${value}" MATCHES "^[0-9a-f]+$")
    message(FATAL_ERROR
      "Invalid ${label} '${value}' in ${HYLE_DEPENDENCY_LOCK_FILE}; "
      "expected ${length} lowercase hexadecimal characters")
  endif()
endfunction()

function(_hyle_dependency_normalized_sha256 path output)
  if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
    message(FATAL_ERROR "Locked file is missing: ${path}")
  endif()
  if(IS_SYMLINK "${path}")
    message(FATAL_ERROR "Locked file must not be a symbolic link: ${path}")
  endif()
  file(READ "${path}" _content)
  string(REPLACE "\r\n" "\n" _content "${_content}")
  string(REPLACE "\r" "\n" _content "${_content}")
  string(SHA256 _sha256 "${_content}")
  set(${output} "${_sha256}" PARENT_SCOPE)
endfunction()

# Hex-encode manifest strings and reject control bytes plus CMake's list
# delimiter.  This keeps the concatenated manifest unambiguous for arbitrary
# Unicode and for names containing the human-readable field delimiter.
function(_hyle_dependency_manifest_hex value label output)
  string(HEX "${value}" _hex)
  if(_hex STREQUAL "")
    message(FATAL_ERROR "Expanded dependency ${label} must not be empty")
  endif()
  string(LENGTH "${_hex}" _hex_length)
  set(_offset 0)
  while(_offset LESS _hex_length)
    string(SUBSTRING "${_hex}" ${_offset} 2 _byte)
    if(_byte MATCHES "^(0[0-9a-f]|1[0-9a-f]|7f|3b)$")
      message(FATAL_ERROR
        "Expanded dependency ${label} contains a control character or ';': "
        "${value}")
    endif()
    math(EXPR _offset "${_offset} + 2")
  endwhile()
  set(${output} "${_hex}" PARENT_SCOPE)
endfunction()

# Canonical identity for a source tree expanded from a locked archive.  Unlike
# the in-tree vendor audit below, file bytes are deliberately not newline
# normalized: an expanded codeload archive is a byte-for-byte supply-chain
# input, not a developer checkout.  Directory entries are included so that
# empty-directory additions/deletions are visible.  Relative symlinks are
# normalized/resolved only for the safety check; absolute and lexically
# escaping targets are rejected before any file is read.  The manifest hashes
# the original portable target spelling, so even a semantically equivalent
# link rewrite (`dir/file` to `dir/./file`) is a detectable mutation.
function(hyle_compute_expanded_source_tree_sha256 root output)
  get_filename_component(_root "${root}" ABSOLUTE)
  file(TO_CMAKE_PATH "${_root}" _root)
  if(NOT IS_DIRECTORY "${_root}" OR IS_SYMLINK "${_root}")
    message(FATAL_ERROR
      "Expanded dependency source must be a real directory: ${_root}")
  endif()

  cmake_policy(PUSH)
  cmake_policy(SET CMP0009 NEW)
  file(GLOB_RECURSE _entries LIST_DIRECTORIES TRUE
       RELATIVE "${_root}" "${_root}/*")
  cmake_policy(POP)
  if(NOT _entries)
    message(FATAL_ERROR "Expanded dependency source is empty: ${_root}")
  endif()
  list(SORT _entries CASE SENSITIVE ORDER ASCENDING)

  set(_manifest "HYLE-EXPANDED-SOURCE-TREE-HEX-v1\n")
  set(_case_folded_entries "")
  set(_has_symbolic_link FALSE)
  foreach(_relative IN LISTS _entries)
    file(TO_CMAKE_PATH "${_relative}" _portable_relative)
    if(_portable_relative MATCHES "(^|/)\\.\\.(/|$)" OR
       IS_ABSOLUTE "${_portable_relative}")
      message(FATAL_ERROR
        "Expanded dependency source has an unsafe path: ${_portable_relative}")
    endif()
    _hyle_dependency_manifest_hex(
      "${_portable_relative}" "path" _relative_hex)
    string(TOLOWER "${_portable_relative}" _case_folded_relative)
    if(_case_folded_relative IN_LIST _case_folded_entries)
      message(FATAL_ERROR
        "Expanded dependency source has a case-colliding path: "
        "${_portable_relative}")
    endif()
    list(APPEND _case_folded_entries "${_case_folded_relative}")

    set(_absolute "${_root}/${_portable_relative}")
    if(IS_SYMLINK "${_absolute}")
      set(_has_symbolic_link TRUE)
      file(READ_SYMLINK "${_absolute}" _link_target)
      file(TO_CMAKE_PATH "${_link_target}" _portable_link_target)
      if(_portable_link_target STREQUAL "" OR
         IS_ABSOLUTE "${_portable_link_target}" OR
         _portable_link_target MATCHES "^[A-Za-z]:/")
        message(FATAL_ERROR
          "Expanded dependency source has an unsafe symbolic link: "
          "${_portable_relative} -> ${_portable_link_target}")
      endif()
      get_filename_component(_link_parent "${_absolute}" DIRECTORY)
      set(_link_target_path "${_portable_link_target}")
      cmake_path(ABSOLUTE_PATH _link_target_path
        BASE_DIRECTORY "${_link_parent}" NORMALIZE
        OUTPUT_VARIABLE _resolved_link_target)
      cmake_path(RELATIVE_PATH _resolved_link_target
        BASE_DIRECTORY "${_root}" OUTPUT_VARIABLE _root_relative_target)
      file(TO_CMAKE_PATH "${_root_relative_target}" _root_relative_target)
      if(IS_ABSOLUTE "${_root_relative_target}" OR
         _root_relative_target STREQUAL ".." OR
         _root_relative_target MATCHES "^\\.\\./")
        message(FATAL_ERROR
          "Expanded dependency source has a symbolic link escaping its root: "
          "${_portable_relative} -> ${_portable_link_target}")
      endif()
      _hyle_dependency_manifest_hex(
        "${_portable_link_target}" "symbolic-link target" _link_target_hex)
      string(APPEND _manifest
        "L|${_relative_hex}|${_link_target_hex}\n")
    elseif(IS_DIRECTORY "${_absolute}")
      string(APPEND _manifest "D|${_relative_hex}\n")
    elseif(EXISTS "${_absolute}")
      file(SHA256 "${_absolute}" _file_sha256)
      string(APPEND _manifest
        "F|${_relative_hex}|${_file_sha256}\n")
    else()
      message(FATAL_ERROR
        "Expanded dependency source contains an unsupported filesystem entry: "
        "${_absolute}")
    endif()
  endforeach()
  string(SHA256 _tree_sha256 "${_manifest}")
  set(${output} "${_tree_sha256}" PARENT_SCOPE)
  # The strict offline preload uses this structural fact when considering its
  # explicit mismatch escape. Exact locked archive trees may contain the safe
  # relative links covered by their digest, but an unverified/mismatching tree
  # must not rely on platform-dependent or multi-hop link resolution.
  if(ARGC GREATER 2)
    set(${ARGV2} "${_has_symbolic_link}" PARENT_SCOPE)
  endif()
endfunction()

# Stable across Windows/Unix checkouts: hash every normalized text file, then
# hash a sorted path|file-sha manifest.  Vendored HYLE snapshots are source and
# license text only; symbolic links are rejected rather than platform-flattened.
function(_hyle_dependency_tree_sha256 root output)
  if(NOT IS_DIRECTORY "${root}")
    message(FATAL_ERROR "Locked vendored directory is missing: ${root}")
  endif()
  file(GLOB_RECURSE _relative_files LIST_DIRECTORIES false
       RELATIVE "${root}" "${root}/*")
  if(NOT _relative_files)
    message(FATAL_ERROR "Locked vendored directory is empty: ${root}")
  endif()
  list(SORT _relative_files)
  set(_manifest "")
  foreach(_relative IN LISTS _relative_files)
    set(_absolute "${root}/${_relative}")
    if(IS_SYMLINK "${_absolute}")
      message(FATAL_ERROR
        "Vendored dependency snapshots must not contain symbolic links: "
        "${_absolute}")
    endif()
    file(TO_CMAKE_PATH "${_relative}" _portable_relative)
    _hyle_dependency_normalized_sha256("${_absolute}" _file_sha256)
    string(APPEND _manifest "${_portable_relative}|${_file_sha256}\n")
  endforeach()
  string(SHA256 _tree_sha256 "${_manifest}")
  set(${output} "${_tree_sha256}" PARENT_SCOPE)
endfunction()

# Identity of the exact HYLE files that implement a modified upstream port.
# The upstream commit describes origin, while this normalized content hash
# describes the reviewed adapted bytes.  They are intentionally separate:
# claiming that a forked port is byte-identical to the upstream tree would be
# false provenance.
function(hyle_compute_integrated_source_content_sha256 root paths output)
  set(_paths ${paths})
  if(NOT _paths)
    message(FATAL_ERROR "Integrated source path set must not be empty")
  endif()
  list(REMOVE_DUPLICATES _paths)
  list(SORT _paths CASE SENSITIVE ORDER ASCENDING)
  set(_manifest "HYLE-INTEGRATED-SOURCE-CONTENT-HEX-v1\n")
  foreach(_path IN LISTS _paths)
    set(_absolute "${root}/${_path}")
    if(NOT EXISTS "${_absolute}" OR IS_SYMLINK "${_absolute}")
      message(FATAL_ERROR
        "Integrated source path is missing or is a symbolic link: ${_absolute}")
    endif()
    _hyle_dependency_manifest_hex("${_path}" "integrated path" _path_hex)
    if(IS_DIRECTORY "${_absolute}")
      _hyle_dependency_tree_sha256("${_absolute}" _content_sha256)
      string(APPEND _manifest "D|${_path_hex}|${_content_sha256}\n")
    else()
      _hyle_dependency_normalized_sha256("${_absolute}" _content_sha256)
      string(APPEND _manifest "F|${_path_hex}|${_content_sha256}\n")
    endif()
  endforeach()
  string(SHA256 _integrated_sha256 "${_manifest}")
  set(${output} "${_integrated_sha256}" PARENT_SCOPE)
endfunction()

function(_hyle_dependency_git_query git checkout label output)
  execute_process(
    COMMAND "${git}" -C "${checkout}" ${ARGN}
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _stdout
    ERROR_VARIABLE _stderr
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE)
  if(NOT _result EQUAL 0)
    message(FATAL_ERROR
      "Cannot verify locked ${label} in '${checkout}' (git exit ${_result}):\n"
      "${_stdout}\n${_stderr}")
  endif()
  set(${output} "${_stdout}" PARENT_SCOPE)
endfunction()

# Public verification entry point for source checkouts supplied outside
# FetchContent (and reused by strict bundle tooling).  A path is acceptable only
# when its commit, root tree, cleanliness and locked license bytes all agree.
function(hyle_verify_locked_git_checkout dependency checkout)
  if(NOT dependency IN_LIST HYLE_DEPENDENCY_NAMES)
    message(FATAL_ERROR
      "'${dependency}' is not present in ${HYLE_DEPENDENCY_LOCK_FILE}")
  endif()
  get_filename_component(_checkout "${checkout}" ABSOLUTE)
  if(NOT IS_DIRECTORY "${_checkout}" OR IS_SYMLINK "${_checkout}" OR
     NOT EXISTS "${_checkout}/.git" OR IS_SYMLINK "${_checkout}/.git")
    message(FATAL_ERROR
      "Locked dependency '${dependency}' must be a real Git checkout: "
      "${_checkout}")
  endif()
  string(TOUPPER "${dependency}" _upper)
  set(_prefix "HYLE_DEP_${_upper}")
  find_package(Git REQUIRED)
  _hyle_dependency_git_query("${GIT_EXECUTABLE}" "${_checkout}"
    "${dependency} HEAD" _head rev-parse --verify HEAD)
  _hyle_dependency_git_query("${GIT_EXECUTABLE}" "${_checkout}"
    "${dependency} root tree" _tree rev-parse --verify "HEAD^{tree}")
  _hyle_dependency_git_query("${GIT_EXECUTABLE}" "${_checkout}"
    "${dependency} worktree" _status
    status --porcelain=v1 --untracked-files=all --ignore-submodules=none)
  _hyle_dependency_git_query("${GIT_EXECUTABLE}" "${_checkout}"
    "${dependency} ignored files" _ignored
    ls-files --others --ignored --exclude-standard)
  if(NOT _head STREQUAL "${${_prefix}_COMMIT}")
    message(FATAL_ERROR
      "Locked dependency '${dependency}' HEAD mismatch.\n"
      "  expected: ${${_prefix}_COMMIT}\n  actual:   ${_head}")
  endif()
  if(NOT _tree STREQUAL "${${_prefix}_GIT_TREE}")
    message(FATAL_ERROR
      "Locked dependency '${dependency}' Git tree mismatch.\n"
      "  expected: ${${_prefix}_GIT_TREE}\n  actual:   ${_tree}")
  endif()
  if(NOT _status STREQUAL "")
    message(FATAL_ERROR
      "Locked dependency '${dependency}' checkout is dirty:\n${_status}")
  endif()
  if(NOT _ignored STREQUAL "")
    message(FATAL_ERROR
      "Locked dependency '${dependency}' checkout contains ignored files. "
      "Release/offline source inputs must not contain hidden build, cache or "
      "generated entries:\n${_ignored}")
  endif()
  foreach(_spec IN LISTS ${_prefix}_LICENSE_FILE_SPECS)
    string(REPLACE "|" ";" _fields "${_spec}")
    list(GET _fields 0 _root)
    list(GET _fields 1 _path)
    list(GET _fields 2 _expected_sha256)
    if(_root STREQUAL "dependency")
      _hyle_dependency_normalized_sha256(
        "${_checkout}/${_path}" _actual_sha256)
      if(NOT _actual_sha256 STREQUAL "${_expected_sha256}")
        message(FATAL_ERROR
          "Locked dependency '${dependency}' license hash mismatch: "
          "${_checkout}/${_path}")
      endif()
    endif()
  endforeach()
endfunction()

function(_hyle_dependency_verify_source_licenses dependency checkout)
  string(TOUPPER "${dependency}" _upper)
  set(_prefix "HYLE_DEP_${_upper}")
  foreach(_spec IN LISTS ${_prefix}_LICENSE_FILE_SPECS)
    string(REPLACE "|" ";" _fields "${_spec}")
    list(GET _fields 0 _root)
    list(GET _fields 1 _path)
    list(GET _fields 2 _expected_sha256)
    if(NOT _root STREQUAL "dependency")
      message(FATAL_ERROR
        "Locked source dependency '${dependency}' has non-dependency license "
        "root '${_root}'")
    endif()
    _hyle_dependency_normalized_sha256(
      "${checkout}/${_path}" _actual_sha256)
    if(NOT _actual_sha256 STREQUAL "${_expected_sha256}")
      message(FATAL_ERROR
        "Locked dependency '${dependency}' license hash mismatch.\n"
        "  path:     ${checkout}/${_path}\n"
        "  expected: ${_expected_sha256}\n"
        "  actual:   ${_actual_sha256}")
    endif()
  endforeach()
endfunction()

# Accept the exact byte tree produced by the official locked archive without
# requiring Git metadata.  This is the normal source-bundle representation.
function(hyle_verify_locked_expanded_source dependency checkout)
  if(NOT dependency IN_LIST HYLE_FETCHCONTENT_DEPENDENCIES AND
     NOT dependency IN_LIST HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES)
    message(FATAL_ERROR
      "'${dependency}' is not a locked expanded-source dependency")
  endif()
  get_filename_component(_checkout "${checkout}" ABSOLUTE)
  if(NOT IS_DIRECTORY "${_checkout}" OR IS_SYMLINK "${_checkout}")
    message(FATAL_ERROR
      "Expanded dependency source must be a real directory: ${_checkout}")
  endif()
  string(TOUPPER "${dependency}" _upper)
  set(_prefix "HYLE_DEP_${_upper}")
  _hyle_dependency_verify_source_licenses("${dependency}" "${_checkout}")
  hyle_compute_expanded_source_tree_sha256(
    "${_checkout}" _actual_tree_sha256)
  if(NOT _actual_tree_sha256 STREQUAL
     "${${_prefix}_EXPANDED_TREE_SHA256}")
    message(FATAL_ERROR
      "Locked dependency '${dependency}' expanded source tree mismatch.\n"
      "  path:     ${_checkout}\n"
      "  expected: ${${_prefix}_EXPANDED_TREE_SHA256}\n"
      "  actual:   ${_actual_tree_sha256}\n"
      "Re-extract the archive whose URL and SHA-256 are recorded in "
      "${HYLE_DEPENDENCY_LOCK_FILE}; do not edit, add or delete entries.")
  endif()
endfunction()

# Unified source-verification entry point.  Git metadata, when present, selects
# the stronger commit/tree/clean proof.  Otherwise the exact expanded-archive
# manifest is required.
function(hyle_verify_locked_source dependency checkout)
  get_filename_component(_checkout "${checkout}" ABSOLUTE)
  if(EXISTS "${_checkout}/.git" OR IS_SYMLINK "${_checkout}/.git")
    hyle_verify_locked_git_checkout("${dependency}" "${_checkout}")
  else()
    hyle_verify_locked_expanded_source("${dependency}" "${_checkout}")
  endif()
endfunction()

# The offline preload may accept a non-matching expanded tree only when it has
# no symbolic links, after its license checks and an explicit unsafe opt-in
# backed by an external attestation. The preload records its result as a GLOBAL property,
# which cannot be forged with a normal `-D` cache argument; ordinary direct
# overrides always reach the strict verifier above.
function(hyle_verify_locked_source_input dependency checkout)
  get_filename_component(_checkout "${checkout}" ABSOLUTE)
  file(TO_CMAKE_PATH "${_checkout}" _checkout)
  cmake_path(NORMAL_PATH _checkout OUTPUT_VARIABLE _checkout)
  string(TOUPPER "${dependency}" _upper)
  set(_attestation_property
      "HYLE_OFFLINE_ATTESTED_UNVERIFIED_SOURCE_${_upper}")
  get_property(_attested_source GLOBAL PROPERTY "${_attestation_property}")
  if(HYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES AND
     "${_attested_source}" STREQUAL "${_checkout}")
    return()
  endif()
  hyle_verify_locked_source("${dependency}" "${_checkout}")
endfunction()

# Standard FetchContent overrides are a supported input surface, but they must
# not bypass the lock simply because they were supplied outside the offline
# preload.  All locked FetchContent consumers use the wrapper macro below.
function(hyle_verify_locked_fetchcontent_override dependency out_source_var)
  if(NOT dependency IN_LIST HYLE_FETCHCONTENT_DEPENDENCIES)
    message(FATAL_ERROR
      "'${dependency}' is not a locked FetchContent dependency")
  endif()
  string(TOUPPER "${dependency}" _upper)
  set(_override_variable "FETCHCONTENT_SOURCE_DIR_${_upper}")
  set(_verified_source "")
  if(DEFINED ${_override_variable} AND
     NOT "${${_override_variable}}" STREQUAL "")
    get_filename_component(_verified_source
      "${${_override_variable}}" ABSOLUTE)
    file(TO_CMAKE_PATH "${_verified_source}" _verified_source)
    cmake_path(NORMAL_PATH _verified_source)
    hyle_verify_locked_source_input(
      "${dependency}" "${_verified_source}")
  endif()
  set(${out_source_var} "${_verified_source}" PARENT_SCOPE)
endfunction()

macro(hyle_fetchcontent_make_available_locked dependency)
  # An explicit standard source override can be authenticated before any
  # dependency CMake executes.  FetchContent's documented "first declaration
  # wins" rule also lets an embedding parent declare the same name earlier,
  # however, and CMake exposes no public declaration-inspection API.  Therefore
  # authenticate the source directory that MakeAvailable actually selected as
  # well.  A trusted parent may execute its predeclared CMake before this check,
  # but an unpinned source can no longer produce a successful HYLE configure or
  # package.  Parents/toolchains are already executable build-code trust roots.
  hyle_verify_locked_fetchcontent_override(
    "${dependency}" _hyle_locked_override_source)
  FetchContent_MakeAvailable(${dependency})
  FetchContent_GetProperties(${dependency}
    SOURCE_DIR _hyle_locked_actual_source)
  if("${_hyle_locked_actual_source}" STREQUAL "")
    message(FATAL_ERROR
      "Locked FetchContent dependency '${dependency}' was made available without "
      "an auditable source directory. A parent declaration, dependency provider "
      "or find-package substitution may not replace HYLE's locked source.")
  endif()
  get_filename_component(_hyle_locked_actual_source
    "${_hyle_locked_actual_source}" ABSOLUTE)
  file(TO_CMAKE_PATH
    "${_hyle_locked_actual_source}" _hyle_locked_actual_source)
  cmake_path(NORMAL_PATH _hyle_locked_actual_source)
  if(NOT "${_hyle_locked_actual_source}" STREQUAL
         "${_hyle_locked_override_source}")
    hyle_verify_locked_source_input(
      "${dependency}" "${_hyle_locked_actual_source}")
  endif()
  unset(_hyle_locked_actual_source)
  unset(_hyle_locked_override_source)
endmacro()

_hyle_dependency_json_get(_hyle_schema_version schema_version)
if(NOT _hyle_schema_version EQUAL 3)
  message(FATAL_ERROR
    "Unsupported HYLE dependency lock schema '${_hyle_schema_version}'")
endif()
_hyle_dependency_json_get(
  _hyle_expanded_tree_algorithm expanded_tree_algorithm)
if(NOT _hyle_expanded_tree_algorithm STREQUAL
   "hyle-path-hex-v1-raw-sha256")
  message(FATAL_ERROR
    "Unsupported expanded source tree algorithm "
    "'${_hyle_expanded_tree_algorithm}'")
endif()
string(JSON _hyle_dependency_count ERROR_VARIABLE _hyle_json_error
       LENGTH "${_hyle_dependency_lock_json}" dependencies)
if(NOT _hyle_json_error STREQUAL "NOTFOUND")
  message(FATAL_ERROR
    "Invalid HYLE dependency lock dependencies array: ${_hyle_json_error}")
endif()
if(_hyle_dependency_count LESS 1)
  message(FATAL_ERROR "HYLE dependency lock contains no dependencies")
endif()

set(HYLE_DEPENDENCY_NAMES "")
set(HYLE_FETCHCONTENT_DEPENDENCIES "")
set(HYLE_IN_TREE_DEPENDENCIES "")
set(HYLE_INTEGRATED_SOURCE_DEPENDENCIES "")
set(HYLE_BENCH_DEPENDENCIES "")
set(HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES "")
math(EXPR _hyle_dependency_last "${_hyle_dependency_count} - 1")
foreach(_hyle_index RANGE 0 ${_hyle_dependency_last})
  _hyle_dependency_json_get(_hyle_name dependencies ${_hyle_index} name)
  _hyle_dependency_json_get(_hyle_class dependencies ${_hyle_index} class)
  _hyle_dependency_json_get(_hyle_repository
                            dependencies ${_hyle_index} repository)
  _hyle_dependency_json_get(_hyle_version dependencies ${_hyle_index} version)
  _hyle_dependency_json_get(_hyle_revision
                            dependencies ${_hyle_index} revision)
  _hyle_dependency_json_get(_hyle_git_tree
                            dependencies ${_hyle_index} git_tree)
  _hyle_dependency_json_get(_hyle_archive_url
                            dependencies ${_hyle_index} archive url)
  _hyle_dependency_json_get(_hyle_archive_sha256
                            dependencies ${_hyle_index} archive sha256)
  _hyle_dependency_json_get(_hyle_license_expression
                            dependencies ${_hyle_index} license expression)

  if(NOT _hyle_name MATCHES "^[a-z][a-z0-9_]*$")
    message(FATAL_ERROR "Invalid dependency name '${_hyle_name}' in lock")
  endif()
  if(_hyle_name IN_LIST HYLE_DEPENDENCY_NAMES)
    message(FATAL_ERROR "Duplicate dependency '${_hyle_name}' in lock")
  endif()
  if(NOT _hyle_class MATCHES
     "^(fetchcontent|in_tree|integrated_source|benchmark|external_checkout)$")
    message(FATAL_ERROR
      "Invalid class '${_hyle_class}' for dependency '${_hyle_name}'")
  endif()
  if(NOT _hyle_repository MATCHES
     "^https://github\\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\\.git$")
    message(FATAL_ERROR
      "Dependency '${_hyle_name}' repository is not a canonical HTTPS "
      "GitHub URL: ${_hyle_repository}")
  endif()
  if(_hyle_version STREQUAL "" OR _hyle_license_expression STREQUAL "")
    message(FATAL_ERROR
      "Dependency '${_hyle_name}' needs non-empty version and license metadata")
  endif()
  _hyle_dependency_require_hex("${_hyle_name} revision" "${_hyle_revision}" 40)
  _hyle_dependency_require_hex("${_hyle_name} Git tree" "${_hyle_git_tree}" 40)
  _hyle_dependency_require_hex(
    "${_hyle_name} archive SHA-256" "${_hyle_archive_sha256}" 64)
  string(REGEX REPLACE "^https://github\\.com/"
         "https://codeload.github.com/" _hyle_expected_archive_url
         "${_hyle_repository}")
  string(REGEX REPLACE "\\.git$" "/tar.gz/${_hyle_revision}"
         _hyle_expected_archive_url "${_hyle_expected_archive_url}")
  if(NOT _hyle_archive_url STREQUAL "${_hyle_expected_archive_url}")
    message(FATAL_ERROR
      "Dependency '${_hyle_name}' archive URL does not exactly match its "
      "repository and revision.\n  expected: ${_hyle_expected_archive_url}\n"
      "  actual:   ${_hyle_archive_url}")
  endif()

  string(TOUPPER "${_hyle_name}" _hyle_upper_name)
  set(_hyle_prefix "HYLE_DEP_${_hyle_upper_name}")
  set(${_hyle_prefix}_CLASS "${_hyle_class}")
  set(${_hyle_prefix}_REPOSITORY "${_hyle_repository}")
  set(${_hyle_prefix}_VERSION "${_hyle_version}")
  set(${_hyle_prefix}_COMMIT "${_hyle_revision}")
  set(${_hyle_prefix}_GIT_TREE "${_hyle_git_tree}")
  set(${_hyle_prefix}_ARCHIVE_URL "${_hyle_archive_url}")
  set(${_hyle_prefix}_ARCHIVE_SHA256 "${_hyle_archive_sha256}")
  set(${_hyle_prefix}_LICENSE_EXPRESSION "${_hyle_license_expression}")

  if(_hyle_class STREQUAL "fetchcontent" OR
     _hyle_class STREQUAL "external_checkout")
    _hyle_dependency_json_get(_hyle_expanded_tree_sha256
                              dependencies ${_hyle_index}
                              expanded_tree_sha256)
    _hyle_dependency_require_hex(
      "${_hyle_name} expanded source tree SHA-256"
      "${_hyle_expanded_tree_sha256}" 64)
    set(${_hyle_prefix}_EXPANDED_TREE_SHA256
        "${_hyle_expanded_tree_sha256}")
  endif()

  string(JSON _hyle_license_count ERROR_VARIABLE _hyle_json_error
         LENGTH "${_hyle_dependency_lock_json}"
         dependencies ${_hyle_index} license files)
  if(NOT _hyle_json_error STREQUAL "NOTFOUND" OR
     _hyle_license_count LESS 1)
    message(FATAL_ERROR
      "Dependency '${_hyle_name}' needs at least one license file")
  endif()
  set(_hyle_license_specs "")
  math(EXPR _hyle_license_last "${_hyle_license_count} - 1")
  foreach(_hyle_license_index RANGE 0 ${_hyle_license_last})
    _hyle_dependency_json_get(
      _hyle_license_root dependencies ${_hyle_index} license files
      ${_hyle_license_index} root)
    _hyle_dependency_json_get(
      _hyle_license_path dependencies ${_hyle_index} license files
      ${_hyle_license_index} path)
    _hyle_dependency_json_get(
      _hyle_license_sha256 dependencies ${_hyle_index} license files
      ${_hyle_license_index} sha256)
    if(NOT _hyle_license_root MATCHES "^(dependency|hyle)$" OR
       _hyle_license_path STREQUAL "" OR
       IS_ABSOLUTE "${_hyle_license_path}" OR
       _hyle_license_path MATCHES "(^|/)\\.\\.(/|$)" OR
       _hyle_license_path MATCHES "[|;]")
      message(FATAL_ERROR
        "Dependency '${_hyle_name}' has an unsafe license file specification")
    endif()
    if(_hyle_class MATCHES "^(in_tree|integrated_source)$")
      set(_hyle_expected_license_root "hyle")
    else()
      set(_hyle_expected_license_root "dependency")
    endif()
    if(NOT _hyle_license_root STREQUAL _hyle_expected_license_root)
      message(FATAL_ERROR
        "Dependency '${_hyle_name}' class '${_hyle_class}' requires license "
        "root '${_hyle_expected_license_root}', got '${_hyle_license_root}'")
    endif()
    _hyle_dependency_require_hex(
      "${_hyle_name} license SHA-256" "${_hyle_license_sha256}" 64)
    list(APPEND _hyle_license_specs
      "${_hyle_license_root}|${_hyle_license_path}|${_hyle_license_sha256}")
  endforeach()
  set(${_hyle_prefix}_LICENSE_FILE_SPECS "${_hyle_license_specs}")

  if(_hyle_class STREQUAL "in_tree")
    _hyle_dependency_json_get(_hyle_vendored_path
                              dependencies ${_hyle_index} vendored path)
    _hyle_dependency_json_get(_hyle_vendored_tree_sha256
                              dependencies ${_hyle_index} vendored tree_sha256)
    if(_hyle_vendored_path STREQUAL "" OR
       IS_ABSOLUTE "${_hyle_vendored_path}" OR
       _hyle_vendored_path MATCHES "(^|/)\\.\\.(/|$)")
      message(FATAL_ERROR
        "Dependency '${_hyle_name}' has unsafe vendored path "
        "'${_hyle_vendored_path}'")
    endif()
    _hyle_dependency_require_hex(
      "${_hyle_name} vendored tree SHA-256"
      "${_hyle_vendored_tree_sha256}" 64)
    set(${_hyle_prefix}_VENDORED_PATH "${_hyle_vendored_path}")
    set(${_hyle_prefix}_VENDORED_TREE_SHA256
        "${_hyle_vendored_tree_sha256}")
    foreach(_hyle_license_spec IN LISTS _hyle_license_specs)
      string(REPLACE "|" ";" _hyle_license_fields
             "${_hyle_license_spec}")
      list(GET _hyle_license_fields 1 _hyle_owned_license_path)
      string(FIND "${_hyle_owned_license_path}"
                  "${_hyle_vendored_path}/" _hyle_license_prefix_index)
      if(NOT _hyle_license_prefix_index EQUAL 0)
        message(FATAL_ERROR
          "In-tree dependency '${_hyle_name}' license path "
          "'${_hyle_owned_license_path}' is outside its vendored path "
          "'${_hyle_vendored_path}'")
      endif()
    endforeach()
    list(APPEND HYLE_IN_TREE_DEPENDENCIES "${_hyle_name}")
  elseif(_hyle_class STREQUAL "integrated_source")
    string(JSON _hyle_integrated_path_count ERROR_VARIABLE _hyle_json_error
           LENGTH "${_hyle_dependency_lock_json}"
           dependencies ${_hyle_index} integrated paths)
    if(NOT _hyle_json_error STREQUAL "NOTFOUND" OR
       _hyle_integrated_path_count LESS 1)
      message(FATAL_ERROR
        "Integrated source '${_hyle_name}' needs at least one HYLE path")
    endif()
    set(_hyle_integrated_paths "")
    math(EXPR _hyle_integrated_path_last
         "${_hyle_integrated_path_count} - 1")
    foreach(_hyle_integrated_path_index
            RANGE 0 ${_hyle_integrated_path_last})
      _hyle_dependency_json_get(
        _hyle_integrated_path dependencies ${_hyle_index} integrated paths
        ${_hyle_integrated_path_index})
      if(_hyle_integrated_path STREQUAL "" OR
         IS_ABSOLUTE "${_hyle_integrated_path}" OR
         _hyle_integrated_path MATCHES "(^|/)\\.\\.(/|$)" OR
         _hyle_integrated_path MATCHES "[|;]")
        message(FATAL_ERROR
          "Integrated source '${_hyle_name}' has unsafe HYLE path "
          "'${_hyle_integrated_path}'")
      endif()
      if(_hyle_integrated_path IN_LIST _hyle_integrated_paths)
        message(FATAL_ERROR
          "Integrated source '${_hyle_name}' has duplicate HYLE path "
          "'${_hyle_integrated_path}'")
      endif()
      list(APPEND _hyle_integrated_paths "${_hyle_integrated_path}")
    endforeach()
    _hyle_dependency_json_get(_hyle_integrated_content_sha256
                              dependencies ${_hyle_index}
                              integrated content_sha256)
    _hyle_dependency_require_hex(
      "${_hyle_name} integrated content SHA-256"
      "${_hyle_integrated_content_sha256}" 64)
    set(${_hyle_prefix}_INTEGRATED_PATHS "${_hyle_integrated_paths}")
    set(${_hyle_prefix}_INTEGRATED_CONTENT_SHA256
        "${_hyle_integrated_content_sha256}")
    foreach(_hyle_license_spec IN LISTS _hyle_license_specs)
      string(REPLACE "|" ";" _hyle_license_fields
             "${_hyle_license_spec}")
      list(GET _hyle_license_fields 1 _hyle_owned_license_path)
      string(FIND "${_hyle_owned_license_path}" "third_party/ported/"
             _hyle_license_prefix_index)
      if(NOT _hyle_license_prefix_index EQUAL 0)
        message(FATAL_ERROR
          "Integrated source '${_hyle_name}' license path "
          "'${_hyle_owned_license_path}' must be under "
          "'third_party/ported/'")
      endif()
    endforeach()
    list(APPEND HYLE_INTEGRATED_SOURCE_DEPENDENCIES "${_hyle_name}")
  elseif(_hyle_class STREQUAL "fetchcontent")
    list(APPEND HYLE_FETCHCONTENT_DEPENDENCIES "${_hyle_name}")
  elseif(_hyle_class STREQUAL "benchmark")
    list(APPEND HYLE_BENCH_DEPENDENCIES "${_hyle_name}")
  else()
    list(APPEND HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES "${_hyle_name}")
  endif()
  list(APPEND HYLE_DEPENDENCY_NAMES "${_hyle_name}")
endforeach()

# When present, the Vulkan stack is one train rather than independently
# drifting pins.  The conditional also permits minimal fixture locks in the
# CMake contract tests.
if(vulkan_headers IN_LIST HYLE_DEPENDENCY_NAMES OR
   volk IN_LIST HYLE_DEPENDENCY_NAMES OR
   glslang IN_LIST HYLE_DEPENDENCY_NAMES)
  foreach(_hyle_vulkan_member IN ITEMS vulkan_headers volk glslang)
    if(NOT _hyle_vulkan_member IN_LIST HYLE_DEPENDENCY_NAMES)
      message(FATAL_ERROR
        "Vulkan dependency lock must contain Vulkan-Headers, volk and glslang")
    endif()
  endforeach()
  if(NOT HYLE_DEP_VULKAN_HEADERS_VERSION STREQUAL HYLE_DEP_VOLK_VERSION OR
     NOT HYLE_DEP_VULKAN_HEADERS_VERSION STREQUAL HYLE_DEP_GLSLANG_VERSION)
    message(FATAL_ERROR
      "Vulkan-Headers, volk and glslang must use one Vulkan SDK version")
  endif()
  set(HYLE_DEP_VULKAN_SDK_VERSION "${HYLE_DEP_VULKAN_HEADERS_VERSION}")
endif()

# Audit only the vendored snapshots that the selected component/feature closure
# consumes.  Projection itself must not make every in-tree vendor a requirement
# for a core-only partial source checkout.  The direct script entry point and
# complete-source packaging call this with the full list.
function(hyle_audit_in_tree_dependencies)
  set(_audit_names ${ARGN})
  list(REMOVE_DUPLICATES _audit_names)
  set(_audit_errors "")
  foreach(_name IN LISTS _audit_names)
    if(NOT _name IN_LIST HYLE_IN_TREE_DEPENDENCIES)
      message(FATAL_ERROR
        "Cannot audit '${_name}': it is not an in-tree dependency in "
        "${HYLE_DEPENDENCY_LOCK_FILE}")
    endif()
    string(TOUPPER "${_name}" _upper_name)
    set(_prefix "HYLE_DEP_${_upper_name}")
    set(_vendored_root
        "${HYLE_DEPENDENCY_SOURCE_ROOT}/${${_prefix}_VENDORED_PATH}")
    if(NOT IS_DIRECTORY "${_vendored_root}")
      string(APPEND _audit_errors
        "\nVendored dependency '${_name}' required by the selected closure "
        "is missing.\n  path: ${_vendored_root}\n")
      continue()
    endif()
    _hyle_dependency_tree_sha256("${_vendored_root}" _actual_tree_sha256)
    if(NOT _actual_tree_sha256 STREQUAL
       "${${_prefix}_VENDORED_TREE_SHA256}")
      string(APPEND _audit_errors
        "\nVendored dependency '${_name}' content does not match the lock.\n"
        "  path:     ${_vendored_root}\n"
        "  expected: ${${_prefix}_VENDORED_TREE_SHA256}\n"
        "  actual:   ${_actual_tree_sha256}\n")
    endif()
    foreach(_license_spec IN LISTS ${_prefix}_LICENSE_FILE_SPECS)
      string(REPLACE "|" ";" _license_fields "${_license_spec}")
      list(GET _license_fields 0 _license_root)
      list(GET _license_fields 1 _license_path)
      list(GET _license_fields 2 _expected_license_sha256)
      if(_license_root STREQUAL "hyle")
        set(_license_absolute
            "${HYLE_DEPENDENCY_SOURCE_ROOT}/${_license_path}")
        if(NOT EXISTS "${_license_absolute}")
          string(APPEND _audit_errors
            "\nVendored dependency '${_name}' license is missing:\n"
            "  path: ${_license_absolute}\n")
          continue()
        endif()
        _hyle_dependency_normalized_sha256(
          "${_license_absolute}" _actual_license_sha256)
        if(NOT _actual_license_sha256 STREQUAL
           "${_expected_license_sha256}")
          string(APPEND _audit_errors
            "\nVendored dependency '${_name}' license hash mismatch:\n"
            "  path:     ${_license_absolute}\n"
            "  expected: ${_expected_license_sha256}\n"
            "  actual:   ${_actual_license_sha256}\n")
        endif()
      endif()
    endforeach()
  endforeach()
  if(NOT _audit_errors STREQUAL "")
    message(FATAL_ERROR
      "HYLE in-tree dependency audit failed:${_audit_errors}"
      "Refresh complete snapshots and their provenance atomically.")
  endif()
endfunction()

# Ported/integrated source is intentionally modified HYLE code, so an upstream
# byte-tree comparison would be false precision.  Audit the declared HYLE
# integration surfaces and their exact selected license copies instead; Git
# tracks the modified product bytes while the lock records the upstream origin.
function(hyle_audit_integrated_source_dependencies)
  set(_audit_names ${ARGN})
  list(REMOVE_DUPLICATES _audit_names)
  set(_audit_errors "")
  foreach(_name IN LISTS _audit_names)
    if(NOT _name IN_LIST HYLE_INTEGRATED_SOURCE_DEPENDENCIES)
      message(FATAL_ERROR
        "Cannot audit '${_name}': it is not an integrated-source dependency "
        "in ${HYLE_DEPENDENCY_LOCK_FILE}")
    endif()
    string(TOUPPER "${_name}" _upper_name)
    set(_prefix "HYLE_DEP_${_upper_name}")
    set(_integrated_paths_valid TRUE)
    foreach(_integrated_path IN LISTS ${_prefix}_INTEGRATED_PATHS)
      set(_integrated_absolute
          "${HYLE_DEPENDENCY_SOURCE_ROOT}/${_integrated_path}")
      if(NOT EXISTS "${_integrated_absolute}" OR
         IS_SYMLINK "${_integrated_absolute}")
        string(APPEND _audit_errors
          "\nIntegrated source '${_name}' is missing a declared HYLE path.\n"
          "  path: ${_integrated_absolute}\n")
        set(_integrated_paths_valid FALSE)
      endif()
    endforeach()
    if(_integrated_paths_valid)
      hyle_compute_integrated_source_content_sha256(
        "${HYLE_DEPENDENCY_SOURCE_ROOT}"
        "${${_prefix}_INTEGRATED_PATHS}" _actual_content_sha256)
      if(NOT _actual_content_sha256 STREQUAL
         "${${_prefix}_INTEGRATED_CONTENT_SHA256}")
        string(APPEND _audit_errors
          "\nIntegrated source '${_name}' adapted content does not match "
          "the lock.\n"
          "  expected: ${${_prefix}_INTEGRATED_CONTENT_SHA256}\n"
          "  actual:   ${_actual_content_sha256}\n")
      endif()
    endif()
    unset(_integrated_paths_valid)
    foreach(_license_spec IN LISTS ${_prefix}_LICENSE_FILE_SPECS)
      string(REPLACE "|" ";" _license_fields "${_license_spec}")
      list(GET _license_fields 0 _license_root)
      list(GET _license_fields 1 _license_path)
      list(GET _license_fields 2 _expected_license_sha256)
      if(NOT _license_root STREQUAL "hyle")
        string(APPEND _audit_errors
          "\nIntegrated source '${_name}' has a non-HYLE license root.\n")
        continue()
      endif()
      set(_license_absolute
          "${HYLE_DEPENDENCY_SOURCE_ROOT}/${_license_path}")
      if(NOT EXISTS "${_license_absolute}")
        string(APPEND _audit_errors
          "\nIntegrated source '${_name}' license is missing:\n"
          "  path: ${_license_absolute}\n")
        continue()
      endif()
      _hyle_dependency_normalized_sha256(
        "${_license_absolute}" _actual_license_sha256)
      if(NOT _actual_license_sha256 STREQUAL
         "${_expected_license_sha256}")
        string(APPEND _audit_errors
          "\nIntegrated source '${_name}' license hash mismatch:\n"
          "  path:     ${_license_absolute}\n"
          "  expected: ${_expected_license_sha256}\n"
          "  actual:   ${_actual_license_sha256}\n")
      endif()
    endforeach()
  endforeach()
  if(NOT _audit_errors STREQUAL "")
    message(FATAL_ERROR
      "HYLE integrated-source dependency audit failed:${_audit_errors}"
      "Refresh provenance, integration paths and license copies atomically.")
  endif()
endfunction()

get_filename_component(_hyle_dependency_reader_file
  "${CMAKE_CURRENT_LIST_FILE}" REALPATH)
if(DEFINED CMAKE_SCRIPT_MODE_FILE)
  get_filename_component(_hyle_dependency_script_file
    "${CMAKE_SCRIPT_MODE_FILE}" REALPATH)
  if("${_hyle_dependency_script_file}" STREQUAL
     "${_hyle_dependency_reader_file}")
    hyle_audit_in_tree_dependencies(${HYLE_IN_TREE_DEPENDENCIES})
    hyle_audit_integrated_source_dependencies(
      ${HYLE_INTEGRATED_SOURCE_DEPENDENCIES})
  endif()
endif()

unset(_hyle_dependency_lock_json)
unset(_hyle_dependency_source_root)
unset(_hyle_dependency_count)
unset(_hyle_dependency_last)
unset(_hyle_json_error)
unset(_hyle_index)
unset(_hyle_name)
unset(_hyle_class)
unset(_hyle_prefix)
unset(_hyle_upper_name)
unset(_hyle_dependency_reader_file)
unset(_hyle_dependency_script_file)
