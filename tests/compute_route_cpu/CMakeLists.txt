hyle_add_header_self_contained_test(compute_route_cpu
  HEADER_DIR compute HEADERS cpu.h)
hyle_add_test(compute_route_cpu SOURCES context_test.cc d5_qwen35_profile_test.cc
  cpu_expanded_gguf_test.cc
  ds4_roundtrip_test.cc
  d5_qwen_gguf_vocabulary_test.cc
  d5_qwen38_official_test.cc
  d5_qwen35_gguf_projection_test.cc
  d5_qwen35_gguf_fragment_test.cc
  DEPS hyle_compute_provider_native hyle_compute_provider_onednn
       hyle_compute_provider_ggml)
target_link_libraries(hyle_compute_route_cpu_test PRIVATE ggml)
target_include_directories(hyle_compute_route_cpu_test SYSTEM PRIVATE
  "${HYLE_GGML_SOURCE_DIR}/src")
hyle_stage_runtime_dependencies(hyle_compute_route_cpu_test)
get_target_property(_hyle_torch_cpu hyle_compute_provider_native HYLE_TORCH_CPU_ENABLED)
if(_hyle_torch_cpu)
  target_compile_definitions(hyle_compute_route_cpu_test PRIVATE HYLE_TEST_TORCH_CPU=1)
endif()
