include_guard(GLOBAL)

# Opt-in, build-tree Windows DLL staging. This file is shared by source
# embedding and installed SDKs; HYLE's own linkage does not describe a
# consumer's complete runtime closure (e.g. a separately compiled kernel DLL).
function(hyle_stage_runtime_dependencies target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "hyle_stage_runtime_dependencies: target '${target}' does not exist")
  endif()
  if(NOT WIN32)
    return()
  endif()
  get_target_property(_existing ${target} HYLE_RUNTIME_STAGING_TARGET)
  if(_existing)
    return()
  endif()

  # No dependency on the executable itself: under CMP0112 NEW,
  # TARGET_FILE_DIR is a location, not an order edge. The stage depends on the
  # actual DLLs, so changing a DLL without changing its import library works.
  set(_stage "hyle_runtime_${target}")
  set(_script "${CMAKE_CURRENT_BINARY_DIR}/hyle-runtime/${target}-$<CONFIG>.cmake")
  set(_content [=[
set(_directory [==[$<TARGET_FILE_DIR:@target@>]==])
set(_dlls [==[$<TARGET_RUNTIME_DLLS:@target@>]==])
file(MAKE_DIRECTORY "${_directory}")
foreach(_dll IN LISTS _dlls)
  get_filename_component(_name "${_dll}" NAME)
  if(NOT _dll STREQUAL "${_directory}/${_name}")
    file(COPY_FILE "${_dll}" "${_directory}/${_name}" ONLY_IF_DIFFERENT)
  endif()
endforeach()
]=])
  string(CONFIGURE "${_content}" _content @ONLY)
  file(GENERATE OUTPUT "${_script}" CONTENT "${_content}")
  add_custom_target(${_stage}
    COMMAND "${CMAKE_COMMAND}" -P "${_script}"
    DEPENDS "$<TARGET_RUNTIME_DLLS:${target}>"
    VERBATIM)
  set_target_properties(${_stage} PROPERTIES FOLDER "hyle/runtime-staging")
  add_dependencies(${target} ${_stage})
  set_property(TARGET ${target} PROPERTY HYLE_RUNTIME_STAGING_TARGET ${_stage})
endfunction()
