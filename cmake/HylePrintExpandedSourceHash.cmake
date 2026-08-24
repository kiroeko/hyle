cmake_minimum_required(VERSION 3.25)

include("${CMAKE_CURRENT_LIST_DIR}/HyleDependencyVersions.cmake")
if(DEFINED HYLE_EXPANDED_SOURCE_SPECS AND
   NOT HYLE_EXPANDED_SOURCE_SPECS STREQUAL "")
  foreach(_hyle_expanded_source_spec IN LISTS HYLE_EXPANDED_SOURCE_SPECS)
    string(FIND "${_hyle_expanded_source_spec}" "|" _separator)
    if(_separator LESS 1)
      message(FATAL_ERROR
        "Each HYLE_EXPANDED_SOURCE_SPECS entry must be name|directory")
    endif()
    string(SUBSTRING "${_hyle_expanded_source_spec}" 0 ${_separator}
                     _name)
    math(EXPR _path_start "${_separator} + 1")
    string(SUBSTRING "${_hyle_expanded_source_spec}" ${_path_start} -1
                     _path)
    if(HYLE_VERIFY_EXPANDED_SOURCES)
      hyle_verify_locked_expanded_source("${_name}" "${_path}")
      message(STATUS "HYLE verified expanded source tree: ${_name}")
    else()
      hyle_compute_expanded_source_tree_sha256(
        "${_path}" _hyle_expanded_source_sha256)
      message(STATUS
        "HYLE expanded source tree ${_name}: ${_hyle_expanded_source_sha256}")
    endif()
  endforeach()
elseif(DEFINED HYLE_EXPANDED_SOURCE_DIR AND
       NOT HYLE_EXPANDED_SOURCE_DIR STREQUAL "")
  hyle_compute_expanded_source_tree_sha256(
    "${HYLE_EXPANDED_SOURCE_DIR}" _hyle_expanded_source_sha256)
  message(STATUS
    "HYLE expanded source tree SHA-256: ${_hyle_expanded_source_sha256}")
else()
  message(FATAL_ERROR
    "Pass -DHYLE_EXPANDED_SOURCE_DIR=<directory> or "
    "-DHYLE_EXPANDED_SOURCE_SPECS=<name|directory;...>")
endif()
