include_guard(GLOBAL)

function(_hyle_headerdeps_targets directory output)
  get_property(_targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
  get_property(_children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
  foreach(_child IN LISTS _children)
    _hyle_headerdeps_targets("${_child}" _child_targets)
    list(APPEND _targets ${_child_targets})
  endforeach()
  set("${output}" "${_targets}" PARENT_SCOPE)
endfunction()

# Legacy localized /showIncludes builds can leave valid objects with no header
# dependencies. A build-local input makes just those existing source scopes
# rebuild normally once; never remove objects or modify source files/times.
function(_hyle_recover_ninja_headerdeps)
  if(NOT MSVC OR NOT CMAKE_GENERATOR MATCHES "^Ninja"
      OR NOT "$ENV{VSLANG}" STREQUAL "1033"
      OR NOT EXISTS "${CMAKE_BINARY_DIR}/build.ninja")
    return()
  endif()
  set(_owner_directory "${HYLE_BINARY_DIR}")
  if(_owner_directory STREQUAL "")
    set(_owner_directory "${CMAKE_CURRENT_BINARY_DIR}")
  endif()
  # Logical CMake children include dedicated dependency binary directories
  # outside this subtree. A path prefix cannot establish ownership: a caller
  # may legitimately place dependency state at the parent build root.
  _hyle_headerdeps_targets("${_owner_directory}" _owned_targets)
  set(_manifests "${CMAKE_BINARY_DIR}/build.ninja")
  foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
    if(EXISTS "${CMAKE_BINARY_DIR}/build-${_config}.ninja")
      list(APPEND _manifests "${CMAKE_BINARY_DIR}/build-${_config}.ninja")
    endif()
  endforeach()
  set(_compdb "")
  set(_deps "")
  foreach(_manifest IN LISTS _manifests)
    # The deps database is shared, but `-t deps` exposes only nodes in this
    # manifest's graph. Union both queries to retain healthy nondefault configs.
    foreach(_tool IN ITEMS compdb deps)
      execute_process(COMMAND "${CMAKE_MAKE_PROGRAM}" -f "${_manifest}" -t "${_tool}"
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        OUTPUT_VARIABLE _database ERROR_VARIABLE _error RESULT_VARIABLE _result
        ENCODING UTF-8)
      if(NOT _result EQUAL 0)
        message(FATAL_ERROR "Cannot inspect Ninja ${_manifest} ${_tool} for header-dependency recovery: ${_error}")
      endif()
      string(APPEND _${_tool} "\n${_database}")
    endforeach()
  endforeach()

  # Parse dependency record headers once, not the potentially large lists of
  # included files. Missing records and zero counts are both untrusted.
  string(REGEX MATCHALL "(^|\n)[^\n]+: #deps [0-9]+,[^\n]*" _records "${_deps}")
  foreach(_record IN LISTS _records)
    string(REGEX REPLACE "^[\r\n]+" "" _record "${_record}")
    if(_record MATCHES "^(.*): #deps ([0-9]+),")
      set(_object "${CMAKE_MATCH_1}")
      set(_count "${CMAKE_MATCH_2}")
      file(TO_CMAKE_PATH "${_object}" _object)
      cmake_path(ABSOLUTE_PATH _object BASE_DIRECTORY "${CMAKE_BINARY_DIR}" NORMALIZE)
      string(SHA256 _key "${_object}")
      set(_headerdeps_${_key} "${_count}")
    endif()
  endforeach()

  # Ninja emits adjacent JSON file/output fields. Decode each small pair with
  # CMake's JSON parser instead of reparsing the full compdb per object. This
  # introduces no Python requirement for independent CMake library consumers.
  string(REGEX MATCHALL
    "\n[ \t]*\"file\"[ \t]*:[^\r\n]*[\r\n]+[ \t]*\"output\"[ \t]*:[^\r\n]*"
    _entries "${_compdb}")
  set(_stamp_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/hyle-headerdeps")
  set(_report "")
  set(_new 0)
  set(_attached 0)
  foreach(_entry IN LISTS _entries)
    string(STRIP "${_entry}" _entry)
    string(REGEX REPLACE ",[ \t]*$" "" _entry "${_entry}")
    string(JSON _source GET "{${_entry}}" file)
    string(JSON _object GET "{${_entry}}" output)
    file(TO_CMAKE_PATH "${_source}" _source)
    file(TO_CMAKE_PATH "${_object}" _object)
    if(NOT _object MATCHES "CMakeFiles/([A-Za-z0-9_.+-]+)\\.dir/.*\\.obj$")
      continue()
    endif()
    set(_target "${CMAKE_MATCH_1}")
    if(NOT _target IN_LIST _owned_targets)
      continue()
    endif()
    cmake_path(ABSOLUTE_PATH _source BASE_DIRECTORY "${CMAKE_BINARY_DIR}" NORMALIZE)
    cmake_path(ABSOLUTE_PATH _object BASE_DIRECTORY "${CMAKE_BINARY_DIR}" NORMALIZE)
    get_source_file_property(_language "${_source}" TARGET_DIRECTORY "${_target}" LANGUAGE)
    if(NOT _language STREQUAL "C" AND NOT _language STREQUAL "CXX")
      continue()
    endif()
    string(SHA256 _key "${_object}")
    if(DEFINED _seen_${_key})
      continue()
    endif()
    set(_seen_${_key} TRUE)
    set(_stamp "${_stamp_dir}/${_key}.stamp")
    if(NOT EXISTS "${_stamp}")
      if(NOT EXISTS "${_object}")
        continue()
      endif()
      if(DEFINED _headerdeps_${_key} AND _headerdeps_${_key} GREATER 0)
        continue()
      endif()
      file(MAKE_DIRECTORY "${_stamp_dir}")
      file(WRITE "${_stamp}" "Hyle Ninja English header-dependency recovery v1\n")
      math(EXPR _new "${_new} + 1")
    endif()
    # Persist the input even after a successful rebuild. Source properties are
    # directory-scoped: other targets in this same scope compiling this source
    # may conservatively rebuild once too; no compiler flags are changed.
    get_property(_inputs SOURCE "${_source}" TARGET_DIRECTORY "${_target}" PROPERTY OBJECT_DEPENDS)
    if(NOT _stamp IN_LIST _inputs)
      set_property(SOURCE "${_source}" TARGET_DIRECTORY "${_target}"
        APPEND PROPERTY OBJECT_DEPENDS "${_stamp}")
    endif()
    math(EXPR _attached "${_attached} + 1")
    foreach(_field IN ITEMS source object stamp target)
      string(REPLACE "\\" "\\\\" _${_field}_json "${_${_field}}")
      string(REPLACE "\"" "\\\"" _${_field}_json "${_${_field}_json}")
    endforeach()
    if(NOT _report STREQUAL "")
      string(APPEND _report ",\n")
    endif()
    string(APPEND _report
      "  {\"source\":\"${_source_json}\",\"object\":\"${_object_json}\",\"target\":\"${_target_json}\",\"stamp\":\"${_stamp_json}\"}")
  endforeach()
  if(_attached GREATER 0)
    file(CONFIGURE OUTPUT "${_stamp_dir}/recovery.json" CONTENT "[\n${_report}\n]\n" @ONLY)
    message(STATUS "HYLE: header-dependency recovery inputs=${_attached}, new stamps=${_new} (same-directory source scopes may include multiple targets)")
  endif()
endfunction()
