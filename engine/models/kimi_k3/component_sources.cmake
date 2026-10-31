include_guard(GLOBAL)

# Model-owned source inventory, callable from the engine or its module test.
# Dependencies are the actual SDK targets of that caller, never a new product.
function(_hyle_add_kimi_k3_component)
  if(TARGET hyle_kimi_k3_component)
    return()
  endif()
  foreach(_dependency IN LISTS ARGN)
    if(NOT TARGET ${_dependency})
      message(FATAL_ERROR "Kimi component requires existing SDK target '${_dependency}'")
    endif()
  endforeach()
  add_library(hyle_kimi_k3_component STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/cold_graph.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/component.cc")
  target_compile_features(hyle_kimi_k3_component PUBLIC cxx_std_23)
  target_link_libraries(hyle_kimi_k3_component PUBLIC ${ARGN})
  if(COMMAND hyle_apply_warnings)
    hyle_apply_warnings(hyle_kimi_k3_component)
  endif()
  if(COMMAND hyle_apply_sanitizer)
    hyle_apply_sanitizer(hyle_kimi_k3_component)
  endif()
  set_target_properties(hyle_kimi_k3_component PROPERTIES FOLDER engine)
endfunction()
