cmake_minimum_required(VERSION 3.25)

foreach(_required IN ITEMS HYLE_VERSION_FILE HYLE_EXACT_VERSION
                           HYLE_INCOMPATIBLE_VERSION)
  if(NOT DEFINED ${_required} OR "${${_required}}" STREQUAL "")
    message(FATAL_ERROR "Pass -D${_required}=...")
  endif()
endforeach()

function(_hyle_check_version requested expected expected_exact)
  string(REPLACE "." ";" _parts "${requested}")
  list(LENGTH _parts PACKAGE_FIND_VERSION_COUNT)
  set(_part_count ${PACKAGE_FIND_VERSION_COUNT})
  while(_part_count LESS 4)
    list(APPEND _parts 0)
    math(EXPR _part_count "${_part_count} + 1")
  endwhile()
  list(GET _parts 0 PACKAGE_FIND_VERSION_MAJOR)
  list(GET _parts 1 PACKAGE_FIND_VERSION_MINOR)
  list(GET _parts 2 PACKAGE_FIND_VERSION_PATCH)
  list(GET _parts 3 PACKAGE_FIND_VERSION_TWEAK)
  set(PACKAGE_FIND_VERSION "${requested}")
  set(PACKAGE_FIND_VERSION_RANGE "")
  set(CMAKE_SIZEOF_VOID_P "")
  unset(PACKAGE_VERSION_COMPATIBLE)
  unset(PACKAGE_VERSION_EXACT)
  include("${HYLE_VERSION_FILE}")
  if(expected AND NOT PACKAGE_VERSION_COMPATIBLE)
    message(FATAL_ERROR
      "installed package unexpectedly rejects compatible version ${requested}")
  elseif(NOT expected AND PACKAGE_VERSION_COMPATIBLE)
    message(FATAL_ERROR
      "installed package unexpectedly accepts incompatible version ${requested}")
  endif()
  if(expected_exact AND NOT PACKAGE_VERSION_EXACT)
    message(FATAL_ERROR
      "installed package does not mark ${requested} as an exact version")
  elseif(NOT expected_exact AND PACKAGE_VERSION_EXACT)
    message(FATAL_ERROR
      "installed package unexpectedly marks ${requested} as exact")
  endif()
endfunction()

_hyle_check_version("${HYLE_EXACT_VERSION}" TRUE TRUE)
if(DEFINED HYLE_COMPATIBLE_VERSION AND
   NOT HYLE_COMPATIBLE_VERSION STREQUAL "")
  _hyle_check_version("${HYLE_COMPATIBLE_VERSION}" TRUE FALSE)
  set(_hyle_compatible_summary "accepts ${HYLE_COMPATIBLE_VERSION}, ")
else()
  set(_hyle_compatible_summary "accepts no non-exact pre-1.0 version, ")
endif()
_hyle_check_version("${HYLE_INCOMPATIBLE_VERSION}" FALSE FALSE)
message(STATUS
  "HYLE package version contract: exact ${HYLE_EXACT_VERSION}, "
  "${_hyle_compatible_summary}"
  "rejects ${HYLE_INCOMPATIBLE_VERSION}")
