include_guard(GLOBAL)

# Existing cold role carriers and sampling-declaration adapter shared by the
# finite model components. No SDK product or default application dependency.
function(_hyle_add_model_components_common)
  if(TARGET hyle_model_components_common)
    return()
  endif()
  add_library(hyle_model_components_common STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_role_source.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/model_sampling_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/math_helpers.cc")
  target_compile_features(hyle_model_components_common PUBLIC cxx_std_23)
  target_link_libraries(hyle_model_components_common PUBLIC ${ARGN})
  if(COMMAND hyle_apply_warnings)
    hyle_apply_warnings(hyle_model_components_common)
  endif()
  if(COMMAND hyle_apply_sanitizer)
    hyle_apply_sanitizer(hyle_model_components_common)
  endif()
  set_target_properties(hyle_model_components_common PROPERTIES FOLDER engine)
endfunction()
