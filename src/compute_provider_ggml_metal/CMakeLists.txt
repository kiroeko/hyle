if(NOT HYLE_COMPUTE_APPLE_ENABLED)
  hyle_add_module(compute_provider_ggml_metal OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_ggml_metal PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()
hyle_add_module(compute_provider_ggml_metal SOURCES native_program.cc)
target_include_directories(hyle_compute_provider_ggml_metal PRIVATE
  "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
target_link_libraries(hyle_compute_provider_ggml_metal PRIVATE ggml)
if(NOT HYLE_STATIC)
  # Upstream's embedded assembly has global symbols independent of C visibility.
  target_link_options(hyle_compute_provider_ggml_metal PRIVATE
    "LINKER:-unexported_symbol,_ggml_*")
endif()
