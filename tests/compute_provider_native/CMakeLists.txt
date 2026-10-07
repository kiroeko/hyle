hyle_add_header_self_contained_test(compute_provider_native
  HEADER_DIR compute
  HEADERS providers/native.h)
add_test(NAME compute_provider_native_default_rule_build_contract
  COMMAND "${CMAKE_COMMAND}"
    -S "${CMAKE_CURRENT_SOURCE_DIR}/default_rule_build_contract"
    -B "${CMAKE_CURRENT_BINARY_DIR}/default-rule-build-contract")
set_tests_properties(compute_provider_native_default_rule_build_contract PROPERTIES
  LABELS "compute_provider_native")
hyle_add_test(compute_provider_native SOURCES
  aten_cast_test.cc
  aten_numeric_test.cc
  aten_transformer_test.cc
  aten_storage_test.cc
  ../compute/operation_test.cc
  cast_test.cc
  elementwise_test.cc
  indexing_test.cc
  matmul_gated_test.cc
  normalization_test.cc
  provider_shape_test.cc
  reduction_test.cc
  transform_test.cc
  transformer_test.cc
  DEPS hyle_compute_backend_cpu)
target_include_directories(hyle_compute_provider_native_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute")
get_target_property(_hyle_torch_cpu hyle_compute_provider_native HYLE_TORCH_CPU_ENABLED)
if(_hyle_torch_cpu)
  target_compile_definitions(hyle_compute_provider_native_test PRIVATE HYLE_TEST_TORCH_CPU=1)
  target_sources(hyle_compute_provider_native_test PRIVATE torch_cpu_test.cc)
  target_link_libraries(hyle_compute_provider_native_test PRIVATE torch)
endif()
