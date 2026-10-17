hyle_add_header_self_contained_test(compute_provider_ggml
  HEADER_DIR compute/providers HEADERS ggml_cpu.h)
if(HYLE_GGML_CPU_ENABLED)
  hyle_add_test(compute_provider_ggml SOURCES cpu_provider_test.cc mature_admission_test.cc
    transformer_fill_test.cc layout_broadcast_test.cc concat_test.cc hadamard_test.cc attention_math_test.cc
    ../compute_ggml_runtime/gallocr_faults.c native_ragged_test.cc native_paged_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_batch hyle_compute_route_cpu hyle_simd)
  target_link_libraries(hyle_compute_provider_ggml_test PRIVATE ggml)
  target_include_directories(hyle_compute_provider_ggml_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute"
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml"
    "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime"
    "${HYLE_GGML_SOURCE_DIR}/src")
  if(MSVC)
    set_source_files_properties(../compute_ggml_runtime/gallocr_faults.c
      PROPERTIES COMPILE_OPTIONS /utf-8)
  endif()
  hyle_stage_runtime(hyle_compute_provider_ggml_test)
endif()
