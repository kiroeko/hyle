if(NOT HYLE_GGML_CPU_ENABLED)
  hyle_add_module(compute_provider_ggml OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_ggml PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()
hyle_add_module(compute_provider_ggml SOURCES ggml_cpu_provider.cc standard_index.cc)
target_include_directories(hyle_compute_provider_ggml PRIVATE
  "${HYLE_SOURCE_DIR}/src/compute"
  "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
target_link_libraries(hyle_compute_provider_ggml PRIVATE ggml)
target_compile_definitions(hyle_compute_provider_ggml PUBLIC HYLE_COMPUTE_GGML_CPU=1)
target_compile_definitions(hyle_compute_provider_ggml PRIVATE
  HYLE_GGML_CPU_PROVIDER_VERSION="${HYLE_GGML_CPU_PROVIDER_VERSION}")
if(HYLE_GGML_CPU_PEER_REVISION STREQUAL "ffa4e8b80930029a35991f94e7c8a93cd67730ab")
  target_compile_definitions(hyle_compute_provider_ggml PRIVATE
    HYLE_GGML_CPU_F32_FLASH=1)
endif()
