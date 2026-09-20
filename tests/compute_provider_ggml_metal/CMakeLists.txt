hyle_ensure_module_test_target(compute_provider_ggml_metal)
if(HYLE_COMPUTE_APPLE_ENABLED)
  hyle_add_test(compute_provider_ggml_metal SOURCES native_program_test.mm)
  set_target_properties(hyle_compute_provider_ggml_metal_test PROPERTIES
    OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON OBJCXX_EXTENSIONS OFF)
  target_link_libraries(hyle_compute_provider_ggml_metal_test PRIVATE
    "-framework Metal" "-framework Foundation")
endif()
