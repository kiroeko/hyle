cmake_minimum_required(VERSION 3.25)

# Initial-cache entry point for an offline source bundle.  Put each dependency
# checkout under deps/<FetchContent-name>/ next to this file, then configure
# HYLE with `cmake -C hyle-offline-deps.cmake ...`.  Directory names are the
# FetchContent_Declare() names, so this mapper needs no duplicated vendor list.
set(_hyle_offline_deps_dir "${CMAKE_CURRENT_LIST_DIR}/deps")
if(NOT IS_DIRECTORY "${_hyle_offline_deps_dir}")
  message(FATAL_ERROR
    "HYLE offline dependency directory does not exist: '${_hyle_offline_deps_dir}'")
endif()

file(GLOB _hyle_offline_entries
  LIST_DIRECTORIES TRUE
  RELATIVE "${_hyle_offline_deps_dir}"
  "${_hyle_offline_deps_dir}/*")
set(_hyle_offline_mapped "")
foreach(_hyle_offline_name IN LISTS _hyle_offline_entries)
  set(_hyle_offline_source
    "${_hyle_offline_deps_dir}/${_hyle_offline_name}")
  if(NOT IS_DIRECTORY "${_hyle_offline_source}")
    continue()
  endif()
  if(NOT _hyle_offline_name MATCHES "^[A-Za-z0-9_+.-]+$")
    message(FATAL_ERROR
      "Invalid offline dependency directory name '${_hyle_offline_name}'")
  endif()
  string(TOUPPER "${_hyle_offline_name}" _hyle_offline_upper)
  set("FETCHCONTENT_SOURCE_DIR_${_hyle_offline_upper}"
    "${_hyle_offline_source}" CACHE PATH
    "Offline source override for ${_hyle_offline_name}")
  list(APPEND _hyle_offline_mapped "${_hyle_offline_name}")
endforeach()

if(NOT _hyle_offline_mapped)
  message(FATAL_ERROR
    "HYLE offline dependency directory contains no source directories: "
    "'${_hyle_offline_deps_dir}'")
endif()
list(SORT _hyle_offline_mapped)
message(STATUS "HYLE offline dependency sources: ${_hyle_offline_mapped}")

unset(_hyle_offline_deps_dir)
unset(_hyle_offline_entries)
unset(_hyle_offline_mapped)
unset(_hyle_offline_name)
unset(_hyle_offline_source)
unset(_hyle_offline_upper)
