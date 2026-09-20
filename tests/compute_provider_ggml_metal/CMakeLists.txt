hyle_ensure_module_test_target(compute_provider_ggml_metal)
if(HYLE_COMPUTE_APPLE_ENABLED)
  hyle_add_test(compute_provider_ggml_metal SOURCES native_program_test.mm)
  set_target_properties(hyle_compute_provider_ggml_metal_test PROPERTIES
    OBJCXX_STANDARD 23 OBJCXX_STANDARD_REQUIRED ON OBJCXX_EXTENSIONS OFF)
  target_link_libraries(hyle_compute_provider_ggml_metal_test PRIVATE
    "-framework Metal" "-framework Foundation")
  if(HYLE_STATIC AND NOT HYLE_TRACE_TRACY)
    target_sources(hyle_compute_provider_ggml_metal_test PRIVATE
      resource_faults.m resource_faults_test.mm)
    target_include_directories(hyle_compute_provider_ggml_metal_test PRIVATE
      "${HYLE_GGML_SOURCE_DIR}/src" "${HYLE_GGML_SOURCE_DIR}/src/ggml-metal")
    target_link_libraries(hyle_compute_provider_ggml_metal_test PRIVATE ggml)
    # Match the vendor build. The source-inclusion fixture contains original
    # ObjC declarations whose warnings are not HYLE's C++ Werror contract.
    set_source_files_properties(resource_faults.m PROPERTIES
      COMPILE_DEFINITIONS GGML_METAL_EMBED_LIBRARY=1 COMPILE_OPTIONS -w)
  endif()
endif()
