hyle_add_header_self_contained_test(compute_provider_ggml
  HEADER_DIR compute/providers HEADERS ggml_cpu.h)
if(HYLE_GGML_CPU_ENABLED)
  hyle_add_test(compute_provider_ggml SOURCES cpu_provider_test.cc
    DEPS hyle_compute_backend_cpu hyle_simd)
endif()
