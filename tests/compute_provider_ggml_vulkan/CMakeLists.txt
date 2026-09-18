if(HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_add_test(compute_provider_ggml_vulkan SOURCES metadata_test.cc region_test.cc native_test.cc work_test.cc storage_test.cc gallocr_faults.c)
  if(MSVC)
    # The C fault fixture includes the original UTF-8 vendor source. C++'s
    # normal build-contract encoding flag does not cover this C translation unit.
    set_source_files_properties(gallocr_faults.c PROPERTIES COMPILE_OPTIONS /utf-8)
  endif()
  target_include_directories(hyle_compute_provider_ggml_vulkan_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_vulkan"
    "${HYLE_GGML_VULKAN_SOURCE_DIR}/src")
  target_link_libraries(hyle_compute_provider_ggml_vulkan_test PRIVATE
    ggml Vulkan-Headers hyle_mem hyle_error hyle_platform)
  hyle_stage_runtime_dependencies(hyle_compute_provider_ggml_vulkan_test)
  add_test(NAME hyle_ggml_vulkan_source_contract_test
    COMMAND "${HYLE_VULKAN_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/source_contract_test.py")
  set_tests_properties(hyle_ggml_vulkan_source_contract_test PROPERTIES
    LABELS "compute_provider_ggml_vulkan")
endif()
