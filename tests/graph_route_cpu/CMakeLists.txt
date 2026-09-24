hyle_add_header_self_contained_test(graph_route_cpu
  HEADER_DIR graph HEADERS cpu.h)
hyle_add_test(graph_route_cpu SOURCES
  ../compute_route_cpu/context_test.cc d5_qwen35_profile_test.cc
  d5_qwen35_gguf_projection_test.cc graph_test.cc
  DEPS hyle_compute_provider_native hyle_compute_provider_onednn
       hyle_graph_provider_onednn hyle_compute_provider_ggml)
target_compile_definitions(hyle_graph_route_cpu_test PRIVATE
  HYLE_TEST_GRAPH_ROUTE_CPU=1)
if(HYLE_GGML_CPU_ENABLED)
  target_link_libraries(hyle_graph_route_cpu_test PRIVATE ggml)
endif()

# The installed/source consumer is also run in the targeted route test suite.
# Sched is a test dependency, not a shipping dependency of the CPU route.
hyle_add_test(graph_route_cpu NAME cpu_executor
  SOURCES ../../samples/package/graph/cpu_executor/main.cc
  DEPS hyle_sched hyle_compute_provider_native hyle_graph_provider_onednn)
