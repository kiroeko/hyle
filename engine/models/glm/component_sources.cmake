include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/../detail/component_sources.cmake")

# Private model code, shared by the application source tree and finite tests.
function(_hyle_add_glm_component)
  if(TARGET hyle_glm_component)
    return()
  endif()
  foreach(dependency IN LISTS ARGN)
    if(NOT TARGET ${dependency})
      message(FATAL_ERROR "GLM component requires existing SDK target '${dependency}'")
    endif()
  endforeach()
  _hyle_add_model_components_common(${ARGN})
  add_library(hyle_glm_component STATIC EXCLUDE_FROM_ALL
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_flash_selection_extent.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_shared_vocabulary.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_moe_storage.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_attention_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_moe_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_gguf_model_types.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_gguf_projection_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_mtp_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_indexer_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_target43_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_target44_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_pro_original_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_pro_target_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_mlx_original_roles.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_target_verifier_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_gguf_compute_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_mlx_affine_projection_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_hyper_connection_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_moe_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_mlx_mtp_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_target43_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_original_target44_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_gguf_f32_cache_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_gguf_mtp_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_pro_mtp_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_pro_target_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_mlx_target_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_gguf_flash_fragment.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_official_index_cache.cc"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/glm_stochastic_prefix_fragment.cc")
  target_compile_features(hyle_glm_component PUBLIC cxx_std_23)
  target_link_libraries(hyle_glm_component PUBLIC ${ARGN} hyle_model_components_common)
  if(COMMAND hyle_apply_warnings)
    hyle_apply_warnings(hyle_glm_component)
  endif()
  if(COMMAND hyle_apply_sanitizer)
    hyle_apply_sanitizer(hyle_glm_component)
  endif()
  set_target_properties(hyle_glm_component PROPERTIES FOLDER engine)
endfunction()
