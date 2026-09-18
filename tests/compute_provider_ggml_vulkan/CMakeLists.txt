if(HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_add_test(compute_provider_ggml_vulkan SOURCES metadata_test.cc region_test.cc)
  target_include_directories(hyle_compute_provider_ggml_vulkan_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_vulkan")
  target_link_libraries(hyle_compute_provider_ggml_vulkan_test PRIVATE
    ggml hyle_mem hyle_error)
  hyle_stage_runtime_dependencies(hyle_compute_provider_ggml_vulkan_test)
  add_test(NAME hyle_ggml_vulkan_source_contract_test
    COMMAND "${HYLE_VULKAN_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract_test.py")
  set_tests_properties(hyle_ggml_vulkan_source_contract_test PROPERTIES
    LABELS "compute_provider_ggml_vulkan")
endif()
