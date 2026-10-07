hyle_add_header_self_contained_test(graph_route_cpu
  HEADER_DIR graph HEADERS cpu.h)
hyle_add_test(graph_route_cpu SOURCES
  ../compute_route_cpu/current_rules_test.cc
  ../compute_route_cpu/cpu_expanded_gguf_test.cc
  ../compute_route_cpu/cpu_mlx_quantized_test.cc
  ../compute_route_cpu/cpu_artifact_decode_test.cc
  ../compute_route_cpu/ds4_roundtrip_test.cc
  ../compute_route_cpu/context_test.cc d5_qwen35_profile_test.cc
  d5_qwen_gguf_vocabulary_test.cc
  d5_qwen38_official_test.cc hybrid_ffn_numeric_test.cc
  d5_qwen35_gguf_projection_test.cc d5_qwen35_gguf_fragment_test.cc graph_test.cc
  DEPS hyle_compute_provider_native hyle_compute_provider_onednn
       hyle_graph_provider_onednn hyle_compute_provider_ggml)
target_compile_definitions(hyle_graph_route_cpu_test PRIVATE
  HYLE_TEST_GRAPH_ROUTE_CPU=1)
get_target_property(_hyle_torch_cpu hyle_compute_provider_native HYLE_TORCH_CPU_ENABLED)
if(_hyle_torch_cpu)
  target_compile_definitions(hyle_graph_route_cpu_test PRIVATE HYLE_TEST_TORCH_CPU=1)
endif()
if(HYLE_GGML_CPU_ENABLED)
  target_link_libraries(hyle_graph_route_cpu_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cpu_test SYSTEM PRIVATE
    "${HYLE_GGML_SOURCE_DIR}/src")
endif()
hyle_stage_runtime_dependencies(hyle_graph_route_cpu_test)

# The installed/source consumer is also run in the targeted route test suite.
# Sched is a test dependency, not a shipping dependency of the CPU route.
hyle_add_test(graph_route_cpu NAME cpu_executor
  SOURCES ../../samples/package/graph/cpu_executor/main.cc
  DEPS hyle_sched hyle_compute_provider_native hyle_graph_provider_onednn)
hyle_stage_runtime_dependencies(hyle_graph_route_cpu_cpu_executor_test)
