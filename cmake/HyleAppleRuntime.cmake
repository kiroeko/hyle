include_guard(GLOBAL)

# Explicit, route-native resource staging. Static MLX loads the metallib next
# to the executable; installed shared MLX also has a copy next to its dylib.
# This helper is identical for source embedding and relocated SDK consumption.
function(hyle_stage_apple_runtime target)
  if(NOT TARGET ${target})
    message(FATAL_ERROR "hyle_stage_apple_runtime: missing target '${target}'")
  endif()
  get_target_property(_type ${target} TYPE)
  if(NOT _type STREQUAL "EXECUTABLE")
    message(FATAL_ERROR "hyle_stage_apple_runtime requires an executable target")
  endif()
  get_target_property(_existing ${target} HYLE_APPLE_RUNTIME_STAGING_TARGET)
  if(_existing)
    return()
  endif()
  set(_producer "")
  if(TARGET hyle::_mlx_native)
    get_filename_component(_asset
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../../mlx.metallib" ABSOLUTE)
    if(NOT EXISTS "${_asset}")
      message(FATAL_ERROR "Hyle Apple SDK is missing runtime resource '${_asset}'")
    endif()
  elseif(TARGET mlx)
    get_target_property(_asset mlx HYLE_APPLE_METALLIB)
    if(NOT _asset)
      message(FATAL_ERROR "MLX is not the HYLE-owned Apple native producer")
    endif()
    set(_producer mlx)
  else()
    message(FATAL_ERROR "hyle_stage_apple_runtime requires the enabled Apple route")
  endif()
  set(_stage "hyle_apple_runtime_${target}")
  add_custom_target(${_stage}
    COMMAND "${CMAKE_COMMAND}" -E make_directory "$<TARGET_FILE_DIR:${target}>"
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "${_asset}" "$<TARGET_FILE_DIR:${target}>/mlx.metallib"
    DEPENDS "${_asset}"
    VERBATIM)
  if(_producer)
    add_dependencies(${_stage} ${_producer})
  endif()
  add_dependencies(${target} ${_stage})
  set_property(TARGET ${target} PROPERTY HYLE_APPLE_RUNTIME_STAGING_TARGET ${_stage})
endfunction()
