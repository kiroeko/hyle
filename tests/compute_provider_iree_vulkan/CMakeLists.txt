hyle_ensure_module_test_target(compute_provider_iree_vulkan)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME hyle_iree_source_contract_test
  COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract_test.py")
set_tests_properties(hyle_iree_source_contract_test PROPERTIES
  LABELS compute_provider_iree_vulkan)
if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_test(compute_provider_iree_vulkan SOURCES native_test.cc queue_plan_test.cc)
  target_include_directories(hyle_compute_provider_iree_vulkan_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_iree_vulkan")
  target_include_directories(hyle_compute_provider_iree_vulkan_test SYSTEM PRIVATE
    "${HYLE_DEP_BUILD_DIR}/iv/src/runtime/src")
  target_link_libraries(hyle_compute_provider_iree_vulkan_test PRIVATE
    hyle_compute_provider_ggml_vulkan hyle_mem hyle_error Vulkan-Headers
    iree_runtime_unified)
  hyle_stage_runtime(hyle_compute_provider_iree_vulkan_test)
endif()
