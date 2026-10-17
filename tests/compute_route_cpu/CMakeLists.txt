hyle_add_header_self_contained_test(compute_route_cpu
  HEADER_DIR compute HEADERS cpu.h)
hyle_add_test(compute_route_cpu SOURCES context_test.cc d5_qwen35_profile_test.cc
  current_rules_test.cc
  cpu_expanded_gguf_test.cc
  cpu_artifact_decode_test.cc
  cpu_mlx_quantized_test.cc
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
hyle_stage_runtime(hyle_compute_route_cpu_test)
get_target_property(_hyle_torch_cpu hyle_compute_provider_native HYLE_TORCH_CPU_ENABLED)
if(_hyle_torch_cpu)
  target_compile_definitions(hyle_compute_route_cpu_test PRIVATE HYLE_TEST_TORCH_CPU=1)
endif()

hyle_add_test(compute_route_cpu NAME sampling_consumer
  SOURCES sampling_consumer.cc DEPS hyle_compute_provider_native hyle_compute_provider_ggml hyle_compute_provider_onednn)
hyle_stage_runtime(hyle_compute_route_cpu_sampling_consumer_test)

# Each scenario gets a fresh process: the startup contract never resets globals.
hyle_add_test(compute_route_cpu NAME thread_runtime NO_AUTORUN
  SOURCES thread_runtime_consumer.cc DEPS hyle_compute_provider_native
  hyle_compute_provider_ggml)
hyle_stage_runtime(hyle_compute_route_cpu_thread_runtime_test)
if(_hyle_torch_cpu)
  set(_hyle_runtime_scenarios startup query-first late-native late-aten race-use race-config metadata-oom)
  hyle_add_test(compute_route_cpu NAME thread_runtime_vendor NO_AUTORUN
    SOURCES thread_runtime_consumer.cc DEPS hyle_compute_provider_native
    hyle_compute_provider_ggml)
  target_compile_definitions(hyle_compute_route_cpu_thread_runtime_vendor_test PRIVATE
    HYLE_G10_TEST_VENDOR=1)
  target_link_libraries(hyle_compute_route_cpu_thread_runtime_vendor_test PRIVATE torch)
  hyle_stage_runtime(hyle_compute_route_cpu_thread_runtime_vendor_test)
  add_test(NAME compute_route_cpu_thread_runtime_foreign
    COMMAND hyle_compute_route_cpu_thread_runtime_vendor_test --foreign)
  set_tests_properties(compute_route_cpu_thread_runtime_foreign PROPERTIES
    LABELS "compute_route_cpu")
  add_test(NAME compute_route_cpu_thread_runtime_qualified_drift
    COMMAND hyle_compute_route_cpu_thread_runtime_vendor_test --qualified-drift)
  set_tests_properties(compute_route_cpu_thread_runtime_qualified_drift PROPERTIES
    LABELS "compute_route_cpu" SKIP_RETURN_CODE 77)
else()
  set(_hyle_runtime_scenarios disabled)
endif()
foreach(_scenario IN LISTS _hyle_runtime_scenarios)
  add_test(NAME compute_route_cpu_thread_runtime_${_scenario}
    COMMAND hyle_compute_route_cpu_thread_runtime_test --${_scenario})
  set_tests_properties(compute_route_cpu_thread_runtime_${_scenario} PROPERTIES
    LABELS "compute_route_cpu")
endforeach()
