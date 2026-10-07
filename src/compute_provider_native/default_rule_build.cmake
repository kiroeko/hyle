# Performance evidence belongs to a build recipe, not just a source revision.
# Unknown recipes retain the ordinary providers without a Native WIN label.
function(hyle_native_default_rule_recipe output)
  set(${output} FALSE PARENT_SCOPE)
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Windows" OR
     NOT HYLE_TARGET_PROCESSOR STREQUAL "x86_64" OR
     NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" OR
     NOT CMAKE_CXX_COMPILER_VERSION STREQUAL "19.50.35728" OR
     NOT HYLE_STATIC OR NOT HYLE_COMPUTE_ONEDNN_ENABLED OR
     NOT HYLE_SANITIZER STREQUAL "off" OR HYLE_TRACE OR
     NOT HYLE_SIMD_EXACT_TARGET STREQUAL "")
    return()
  endif()

  string(REGEX REPLACE "[ \t\r\n]+" " " _flags "${CMAKE_CXX_FLAGS}")
  string(STRIP "${_flags}" _flags)
  string(REGEX REPLACE "[ \t\r\n]+" " " _release "${CMAKE_CXX_FLAGS_RELEASE}")
  string(STRIP "${_release}" _release)
  if(NOT _flags STREQUAL "/DWIN32 /D_WINDOWS /EHsc" OR
     NOT _release STREQUAL "/O2 /Ob2 /DNDEBUG" OR
     CMAKE_INTERPROCEDURAL_OPTIMIZATION OR
     CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE)
    return()
  endif()
  if(CMAKE_MSVC_RUNTIME_LIBRARY AND
     NOT CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreadedDLL" AND
     NOT CMAKE_MSVC_RUNTIME_LIBRARY STREQUAL "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    return()
  endif()

  # Embedding-directory flags can override the recipe even with matching cache
  # values. Do not infer equivalence for arbitrary caller compile options.
  get_directory_property(_options COMPILE_OPTIONS)
  get_directory_property(_definitions COMPILE_DEFINITIONS)
  if(_options OR _definitions)
    return()
  endif()
  foreach(_property IN ITEMS INTERPROCEDURAL_OPTIMIZATION INTERPROCEDURAL_OPTIMIZATION_RELEASE)
    get_target_property(_ipo hyle_compute_provider_native ${_property})
    if(_ipo)
      return()
    endif()
  endforeach()
  set(${output} TRUE PARENT_SCOPE)
endfunction()
