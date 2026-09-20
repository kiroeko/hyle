hyle_ensure_module_test_target(compute_provider_iree_vulkan)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME hyle_iree_source_contract_test
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract_test.py")
set_tests_properties(hyle_iree_source_contract_test PROPERTIES
  LABELS compute_provider_iree_vulkan)
if(HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_add_test(compute_provider_iree_vulkan SOURCES native_test.cc)
  target_include_directories(hyle_compute_provider_iree_vulkan_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_iree_vulkan")
  target_link_libraries(hyle_compute_provider_iree_vulkan_test PRIVATE
    hyle_compute_provider_ggml_vulkan hyle_mem hyle_error Vulkan-Headers)
  hyle_stage_runtime_dependencies(hyle_compute_provider_iree_vulkan_test)
endif()
