include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/../detail/component_sources.cmake")

function(_hyle_add_deepseek_component)
  if(TARGET hyle_deepseek_component)
    return()
  endif()
  foreach(dependency IN LISTS ARGN)
    if(NOT TARGET ${dependency})
      message(FATAL_ERROR "DeepSeek component requires existing SDK target '${dependency}'")
    endif()
  endforeach()
  _hyle_add_model_components_common(${ARGN})
  add_library(hyle_deepseek_component STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_block5_policy.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_stored_projection_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_target_binding.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_target_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_shared_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_attention_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_expert_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_hc_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_formal_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_vocabulary_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_formal_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_draft_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_target_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_draft_program.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_original_target_program.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_stochastic_head.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_component_math.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/dspark_stored_projection_graph.cc")
  target_compile_features(hyle_deepseek_component PUBLIC cxx_std_23)
  target_link_libraries(hyle_deepseek_component PUBLIC ${ARGN} hyle_model_components_common)
  if(COMMAND hyle_apply_warnings)
    hyle_apply_warnings(hyle_deepseek_component)
  endif()
  if(COMMAND hyle_apply_sanitizer)
    hyle_apply_sanitizer(hyle_deepseek_component)
  endif()
  set_target_properties(hyle_deepseek_component PROPERTIES FOLDER engine)
endfunction()
