hyle_add_header_self_contained_test(compute_route_cpu
  HEADER_DIR compute HEADERS cpu.h)
hyle_add_test(compute_route_cpu SOURCES context_test.cc d5_qwen35_profile_test.cc
  DEPS hyle_compute_provider_native hyle_compute_provider_onednn
       hyle_compute_provider_ggml)
target_link_libraries(hyle_compute_route_cpu_test PRIVATE ggml)
