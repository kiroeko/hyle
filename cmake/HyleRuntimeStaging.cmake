include_guard(GLOBAL)

# Some native SDKs omit loader-only DLLs from their imported target graph.
# Associate those files with real owners, rather than copying them into every
# executable merely because the SDK was discovered during configuration.
function(hyle_register_runtime_files)
  cmake_parse_arguments(ARG "" "" "TARGETS;FILES" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR NOT ARG_TARGETS OR NOT ARG_FILES)
    message(FATAL_ERROR "hyle_register_runtime_files requires TARGETS and FILES")
  endif()
  foreach(_owner IN LISTS ARG_TARGETS)
    string(SHA256 _key "${_owner}")
    set_property(GLOBAL APPEND PROPERTY "HYLE_RUNTIME_FILES_${_key}" ${ARG_FILES})
  endforeach()
endfunction()

function(_hyle_collect_runtime_files target output)
  # CMake configure is serialized and this walker does not invoke callbacks.
  # Reset per collection: targets/runtime assets may change before a later call.
  set_property(GLOBAL PROPERTY HYLE_RUNTIME_WALK_SEEN "")
  _hyle_runtime_files_walk("${target}" "1" "" _files)
  list(REMOVE_DUPLICATES _files)
  set(${output} "${_files}" PARENT_SCOPE)
endfunction()

function(_hyle_runtime_files_walk owner condition ancestors output)
  set(_files "")
  if(NOT TARGET "${owner}" OR owner IN_LIST ancestors)
    set(${output} "" PARENT_SCOPE)
    return()
  endif()
  string(SHA256 _visit "${owner}|${condition}")
  get_property(_seen GLOBAL PROPERTY HYLE_RUNTIME_WALK_SEEN)
  if(_visit IN_LIST _seen)
    set(${output} "" PARENT_SCOPE)
    return()
  endif()
  set_property(GLOBAL APPEND PROPERTY HYLE_RUNTIME_WALK_SEEN "${_visit}")
  list(APPEND ancestors "${owner}")
  string(SHA256 _key "${owner}")
  get_property(_extra GLOBAL PROPERTY "HYLE_RUNTIME_FILES_${_key}")
  foreach(_file IN LISTS _extra)
    list(APPEND _files "$<${condition}:${_file}>")
  endforeach()
  get_target_property(_alias "${owner}" ALIASED_TARGET)
  if(_alias)
    _hyle_runtime_files_walk("${_alias}" "${condition}" "${ancestors}" _alias_files)
    list(APPEND _files ${_alias_files})
  endif()
  foreach(_property IN ITEMS LINK_LIBRARIES INTERFACE_LINK_LIBRARIES
      IMPORTED_LINK_DEPENDENT_LIBRARIES IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE
      IMPORTED_LINK_DEPENDENT_LIBRARIES_DEBUG IMPORTED_LINK_DEPENDENT_LIBRARIES_RELWITHDEBINFO)
    get_target_property(_links "${owner}" "${_property}")
    list(REMOVE_DUPLICATES _links)
    set(_property_condition "${condition}")
    if(_property MATCHES "_(RELEASE|DEBUG|RELWITHDEBINFO)$")
      set(_property_condition "$<AND:${condition},$<CONFIG:${CMAKE_MATCH_1}>>")
    endif()
    foreach(_link IN LISTS _links)
      # These single-target wrappers have no conditional runtime branch in the
      # current build. Keep genuine config/boolean expressions below intact.
      set(_literal "${_link}")
      while(_literal MATCHES "^\\$<(LINK_ONLY|BUILD_INTERFACE|1):(.*)>$")
        set(_literal "${CMAKE_MATCH_2}")
      endwhile()
      if(TARGET "${_literal}")
        set(_link "${_literal}")
      endif()
      if(TARGET "${_link}")
        _hyle_runtime_files_walk("${_link}" "${_property_condition}" "${ancestors}" _linked_files)
        list(APPEND _files ${_linked_files})
      else()
        # Discover possible identities, then preserve the actual evaluated
        # edge. An inactive config/boolean branch must not stage its files.
        # LINK_ONLY suppresses usage requirements, not runtime linkage.
        string(REPLACE "$<LINK_ONLY:" "$<1:" _evaluated_link "${_link}")
        string(REGEX MATCHALL "[A-Za-z0-9_.+-]+(::[A-Za-z0-9_.+-]+)*" _names "${_link}")
        list(REMOVE_DUPLICATES _names)
        foreach(_name IN LISTS _names)
          if(TARGET "${_name}")
            set(_edge "$<IN_LIST:${_name},$<TARGET_GENEX_EVAL:${owner},${_evaluated_link}>>")
            _hyle_runtime_files_walk("${_name}" "$<AND:${_property_condition},${_edge}>" "${ancestors}" _linked_files)
            list(APPEND _files ${_linked_files})
          endif()
        endforeach()
      endif()
    endforeach()
  endforeach()
  set(${output} "${_files}" PARENT_SCOPE)
endfunction()

# Opt-in Windows DLL staging and Linux SDK loader paths. This file is shared by
# source embedding and installed SDKs; HYLE's own linkage does not describe a
# consumer's complete runtime closure (e.g. a separately compiled kernel DLL).
function(hyle_stage_runtime_dependencies target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "hyle_stage_runtime_dependencies: target '${target}' does not exist")
  endif()
  get_target_property(_existing ${target} HYLE_RUNTIME_STAGING_TARGET)
  if(_existing)
    return()
  endif()

  if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|Darwin)$")
    _hyle_collect_runtime_files("${target}" _runtime_files)
    if(_runtime_files)
      set(_runtime_paths "")
      foreach(_file IN LISTS _runtime_files)
        list(APPEND _runtime_paths "$<PATH:GET_PARENT_PATH,${_file}>")
      endforeach()
      set_property(TARGET ${target} APPEND PROPERTY BUILD_RPATH ${_runtime_paths})
      set_property(TARGET ${target} APPEND PROPERTY INSTALL_RPATH ${_runtime_paths})
      # RUNPATH is not transitive: the HYLE shared adapter, rather than the
      # executable, needs torch_cpu. Opt-in RPATH also covers its SDK-local
      # dependencies, without making a second SDK copy for every application.
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        target_link_options(${target} PRIVATE
          "$<$<BOOL:$<JOIN:${_runtime_paths},>>:LINKER:--disable-new-dtags>")
      endif()
    endif()
    set_property(TARGET ${target} PROPERTY HYLE_RUNTIME_STAGING_TARGET TRUE)
    return()
  elseif(NOT WIN32)
    return()
  endif()

  set(_stage "hyle_runtime_${target}")
  _hyle_collect_runtime_files("${target}" _extra_runtime_files)
  set(_script "${CMAKE_CURRENT_BINARY_DIR}/hyle-runtime/${target}-$<CONFIG>.cmake")
  set(_content [=[
set(_directory [==[$<TARGET_FILE_DIR:@target@>]==])
set(_dlls [==[$<TARGET_RUNTIME_DLLS:@target@>]==])
set(_extra_dlls [==[@_extra_runtime_files@]==])
list(APPEND _dlls ${_extra_dlls})
list(REMOVE_DUPLICATES _dlls)
file(MAKE_DIRECTORY "${_directory}")
foreach(_dll IN LISTS _dlls)
  get_filename_component(_name "${_dll}" NAME)
  if(NOT _dll STREQUAL "${_directory}/${_name}")
    file(COPY_FILE "${_dll}" "${_directory}/${_name}" ONLY_IF_DIFFERENT)
  endif()
endforeach()
]=])
  string(CONFIGURE "${_content}" _content @ONLY)
  file(GENERATE OUTPUT "${_script}" CONTENT "${_content}" TARGET ${target})
  if(CMAKE_GENERATOR MATCHES "^Visual Studio ")
    # MSBuild's project references finish before this event. File dependencies
    # on transitive DLL paths alone do not establish cross-directory project
    # order. PRE_BUILD also runs when the linker keeps an unchanged EXE.
    add_custom_command(TARGET ${target} PRE_BUILD
      COMMAND "${CMAKE_COMMAND}" -P "${_script}"
      VERBATIM)
    set_property(TARGET ${target} PROPERTY HYLE_RUNTIME_STAGING_TARGET TRUE)
    return()
  endif()

  # No dependency on the executable itself: under CMP0112 NEW,
  # TARGET_FILE_DIR is a location, not an order edge. The stage depends on the
  # actual DLLs, so changing a DLL without changing its import library works.
  add_custom_target(${_stage}
    COMMAND "${CMAKE_COMMAND}" -P "${_script}"
    DEPENDS "$<TARGET_RUNTIME_DLLS:${target}>" ${_extra_runtime_files}
    VERBATIM)
  set_target_properties(${_stage} PROPERTIES FOLDER "hyle/runtime-staging")
  add_dependencies(${target} ${_stage})
  set_property(TARGET ${target} PROPERTY HYLE_RUNTIME_STAGING_TARGET ${_stage})
endfunction()
