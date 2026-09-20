if(NOT HYLE_GGML_CPU_ENABLED)
  hyle_add_module(compute_provider_ggml OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_ggml PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()
hyle_add_module(compute_provider_ggml SOURCES ggml_cpu_provider.cc)
target_include_directories(hyle_compute_provider_ggml PRIVATE
  "${HYLE_SOURCE_DIR}/src/compute"
  "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
target_link_libraries(hyle_compute_provider_ggml PRIVATE ggml)
target_compile_definitions(hyle_compute_provider_ggml PUBLIC HYLE_COMPUTE_GGML_CPU=1)
