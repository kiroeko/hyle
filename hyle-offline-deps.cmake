cmake_minimum_required(VERSION 3.25)

# Initial-cache entry point for a verified, network-disconnected source bundle.
# Put exact clean Git checkouts or untouched official-archive expansions under
# deps/<FetchContent-name>/ next to this file, then configure HYLE with
# `cmake -C hyle-offline-deps.cmake ...`.
include("${CMAKE_CURRENT_LIST_DIR}/cmake/HyleDependencyVersions.cmake")

set(HYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES OFF CACHE BOOL
  "Permit an externally attested expanded-tree mismatch (unsafe)")
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL
  "Never download a missing HYLE FetchContent dependency" FORCE)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON CACHE BOOL
  "Never contact remotes while updating HYLE FetchContent dependencies" FORCE)

set(_hyle_offline_deps_dir "${CMAKE_CURRENT_LIST_DIR}/deps")

function(_hyle_offline_validate_licenses name source prefix)
  foreach(_spec IN LISTS ${prefix}_LICENSE_FILE_SPECS)
    string(REPLACE "|" ";" _fields "${_spec}")
    list(GET _fields 0 _root)
    list(GET _fields 1 _path)
    list(GET _fields 2 _expected_sha256)
    if(NOT _root STREQUAL "dependency")
      message(FATAL_ERROR
        "FetchContent dependency '${name}' has non-dependency license root "
        "'${_root}' in the lock")
    endif()
    set(_absolute "${source}/${_path}")
    _hyle_dependency_normalized_sha256("${_absolute}" _actual_sha256)
    if(NOT _actual_sha256 STREQUAL "${_expected_sha256}")
      message(FATAL_ERROR
        "Offline dependency '${name}' license hash mismatch.\n"
        "  path:     ${_absolute}\n"
        "  expected: ${_expected_sha256}\n"
        "  actual:   ${_actual_sha256}")
    endif()
  endforeach()
endfunction()

set(_hyle_offline_entries "")
if(IS_DIRECTORY "${_hyle_offline_deps_dir}")
  file(GLOB _hyle_offline_entries
    LIST_DIRECTORIES TRUE
    RELATIVE "${_hyle_offline_deps_dir}"
    "${_hyle_offline_deps_dir}/*")
endif()
set(_hyle_offline_mapped "")
set(_hyle_offline_attestation_script_lines "")
set(_hyle_offline_allowed_names
    ${HYLE_FETCHCONTENT_DEPENDENCIES} ${HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES})
foreach(_hyle_offline_name IN LISTS _hyle_offline_allowed_names)
  string(TOUPPER "${_hyle_offline_name}" _hyle_offline_upper)
  set_property(GLOBAL PROPERTY
    "HYLE_OFFLINE_ATTESTED_UNVERIFIED_SOURCE_${_hyle_offline_upper}" "")
  # Clear the obsolete cache-shaped token as well, so an old build tree or a
  # command-line spoof cannot retain authority after this contract change.
  unset("HYLE_OFFLINE_ATTESTED_UNVERIFIED_SOURCE_${_hyle_offline_upper}"
        CACHE)
endforeach()
foreach(_hyle_offline_name IN LISTS _hyle_offline_entries)
  set(_hyle_offline_source
      "${_hyle_offline_deps_dir}/${_hyle_offline_name}")
  if(NOT IS_DIRECTORY "${_hyle_offline_source}" OR
     IS_SYMLINK "${_hyle_offline_source}")
    message(FATAL_ERROR
      "Every deps/ entry must be a real dependency directory: "
      "'${_hyle_offline_source}'")
  endif()
  if(NOT _hyle_offline_name IN_LIST _hyle_offline_allowed_names)
    message(FATAL_ERROR
      "Offline directory '${_hyle_offline_name}' is not a product source name "
      "allowed by ${HYLE_DEPENDENCY_LOCK_FILE}. Allowed names: "
      "${_hyle_offline_allowed_names}")
  endif()
  if(_hyle_offline_name IN_LIST _hyle_offline_mapped)
    message(FATAL_ERROR
      "Duplicate offline dependency directory '${_hyle_offline_name}'")
  endif()

  string(TOUPPER "${_hyle_offline_name}" _hyle_offline_upper)
  set(_hyle_offline_prefix "HYLE_DEP_${_hyle_offline_upper}")
  _hyle_offline_validate_licenses(
    "${_hyle_offline_name}" "${_hyle_offline_source}"
    "${_hyle_offline_prefix}")
  if(EXISTS "${_hyle_offline_source}/.git" OR
     IS_SYMLINK "${_hyle_offline_source}/.git")
    hyle_verify_locked_git_checkout(
      "${_hyle_offline_name}" "${_hyle_offline_source}")
  else()
    # The structural hash routine rejects direct absolute/lexically escaping
    # links and also reports whether the tree contains any link. Exact locked
    # archive trees may retain their digest-covered safe relative links; the
    # explicit mismatch escape rejects all links because lexical checks cannot
    # prove that a target reached through another symlink remains under root.
    hyle_compute_expanded_source_tree_sha256(
      "${_hyle_offline_source}" _hyle_offline_expanded_tree_sha256
      _hyle_offline_expanded_has_symbolic_link)
    if(NOT _hyle_offline_expanded_tree_sha256 STREQUAL
       "${${_hyle_offline_prefix}_EXPANDED_TREE_SHA256}")
      if(HYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES)
        if(_hyle_offline_expanded_has_symbolic_link)
          message(FATAL_ERROR
            "Offline dependency '${_hyle_offline_name}' uses the explicit "
            "unverified expanded-source escape but contains symbolic links. "
            "Only an exact lock-matching archive tree may contain digest-"
            "covered safe relative links; use a link-free attested tree or "
            "an exact clean Git checkout.")
        endif()
        file(TO_CMAKE_PATH "${_hyle_offline_source}"
             _hyle_offline_attested_source)
        cmake_path(NORMAL_PATH _hyle_offline_attested_source
                   OUTPUT_VARIABLE _hyle_offline_attested_source)
        set_property(GLOBAL PROPERTY
          "HYLE_OFFLINE_ATTESTED_UNVERIFIED_SOURCE_${_hyle_offline_upper}"
          "${_hyle_offline_attested_source}")
        # Initial-cache scripts run before project() and their GLOBAL
        # properties do not survive project initialization. Emit a build-tree
        # top-level include that restores only the literal paths this preload
        # just inspected. Ordinary -D cache arguments cannot synthesize this
        # generated code. As with all CMake security gates, a caller already
        # authorized to execute arbitrary project include code is trusted.
        if(_hyle_offline_attested_source MATCHES "[;\r\n]" OR
           _hyle_offline_attested_source MATCHES "]====]")
          message(FATAL_ERROR
            "Unsafe offline attestation path cannot be represented: "
            "${_hyle_offline_attested_source}")
        endif()
        string(APPEND _hyle_offline_attestation_script_lines
          "set_property(GLOBAL PROPERTY "
          "[[HYLE_OFFLINE_ATTESTED_UNVERIFIED_SOURCE_${_hyle_offline_upper}]] "
          "[====[${_hyle_offline_attested_source}]====])\n")
        message(WARNING
          "Offline dependency '${_hyle_offline_name}' does not match the "
          "locked official-archive expansion. The explicit unsafe escape is "
          "enabled; the distributor must retain independent provenance.\n"
          "  expected: ${${_hyle_offline_prefix}_EXPANDED_TREE_SHA256}\n"
          "  actual:   ${_hyle_offline_expanded_tree_sha256}")
      else()
        message(FATAL_ERROR
          "Offline dependency '${_hyle_offline_name}' expanded source tree "
          "mismatch.\n"
          "  path:     ${_hyle_offline_source}\n"
          "  expected: ${${_hyle_offline_prefix}_EXPANDED_TREE_SHA256}\n"
          "  actual:   ${_hyle_offline_expanded_tree_sha256}\n"
          "Re-extract the exact locked archive without editing, adding or "
          "deleting entries. HYLE_OFFLINE_ALLOW_UNVERIFIED_EXPANDED_SOURCES="
          "ON is reserved for independently attested external sources.")
      endif()
    endif()
  endif()
  if(_hyle_offline_name IN_LIST HYLE_FETCHCONTENT_DEPENDENCIES)
    set("FETCHCONTENT_SOURCE_DIR_${_hyle_offline_upper}"
        "${_hyle_offline_source}" CACHE PATH
        "Verified offline source override for ${_hyle_offline_name}" FORCE)
  elseif(_hyle_offline_name STREQUAL "tracy")
    set(HYLE_TRACY_DIR "${_hyle_offline_source}" CACHE PATH
        "Verified offline Tracy source checkout" FORCE)
  else()
    message(FATAL_ERROR
      "Offline mapping is not implemented for locked external checkout "
      "'${_hyle_offline_name}'")
  endif()
  list(APPEND _hyle_offline_mapped "${_hyle_offline_name}")
endforeach()

if(NOT _hyle_offline_attestation_script_lines STREQUAL "")
  set(_hyle_offline_attestation_init
      "${CMAKE_BINARY_DIR}/CMakeFiles/HyleOfflineAttestations.cmake")
  file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/CMakeFiles")
  file(WRITE "${_hyle_offline_attestation_init}"
    "# Generated by the verified HYLE offline preload.\n"
    "${_hyle_offline_attestation_script_lines}")
  set(_hyle_offline_top_level_includes
      ${CMAKE_PROJECT_TOP_LEVEL_INCLUDES})
  list(APPEND _hyle_offline_top_level_includes
       "${_hyle_offline_attestation_init}")
  list(REMOVE_DUPLICATES _hyle_offline_top_level_includes)
  set(CMAKE_PROJECT_TOP_LEVEL_INCLUDES
      "${_hyle_offline_top_level_includes}" CACHE STRING
      "Top-level includes required by the verified HYLE offline preload"
      FORCE)
endif()

# Predeclare a deterministic local path even for absent locked inputs.  A
# source-free component closure never touches it; selecting that dependency
# makes FetchContent fail clearly on the missing bundle directory instead of
# treating FULLY_DISCONNECTED as an empty successful population.
foreach(_hyle_offline_name IN LISTS HYLE_FETCHCONTENT_DEPENDENCIES)
  if(NOT _hyle_offline_name IN_LIST _hyle_offline_mapped)
    string(TOUPPER "${_hyle_offline_name}" _hyle_offline_upper)
    set("FETCHCONTENT_SOURCE_DIR_${_hyle_offline_upper}"
        "${_hyle_offline_deps_dir}/${_hyle_offline_name}" CACHE PATH
        "Expected offline source for ${_hyle_offline_name}" FORCE)
  endif()
endforeach()
if(tracy IN_LIST HYLE_EXTERNAL_CHECKOUT_DEPENDENCIES AND
   NOT tracy IN_LIST _hyle_offline_mapped)
  set(HYLE_TRACY_DIR "${_hyle_offline_deps_dir}/tracy" CACHE PATH
      "Expected offline Tracy source checkout" FORCE)
endif()

if(_hyle_offline_mapped)
  list(SORT _hyle_offline_mapped)
  message(STATUS
    "HYLE verified offline dependency sources: ${_hyle_offline_mapped}")
else()
  message(STATUS
    "HYLE offline dependency sources: none (valid for a source-free "
    "component/feature selection)")
endif()
message(STATUS
  "HYLE offline network contract: FETCHCONTENT_FULLY_DISCONNECTED=ON")

unset(_hyle_offline_deps_dir)
unset(_hyle_offline_entries)
unset(_hyle_offline_mapped)
unset(_hyle_offline_name)
unset(_hyle_offline_source)
unset(_hyle_offline_upper)
unset(_hyle_offline_prefix)
unset(_hyle_offline_allowed_names)
unset(_hyle_offline_expanded_tree_sha256)
unset(_hyle_offline_expanded_has_symbolic_link)
unset(_hyle_offline_attested_source)
unset(_hyle_offline_attestation_script_lines)
unset(_hyle_offline_attestation_init)
unset(_hyle_offline_top_level_includes)
