hyle_add_header_self_contained_test(compute_provider_native
  HEADER_DIR compute
  HEADERS providers/native.h)
hyle_add_test(compute_provider_native SOURCES
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
  target_sources(hyle_compute_provider_native_test PRIVATE torch_cpu_test.cc)
  target_link_libraries(hyle_compute_provider_native_test PRIVATE torch)
endif()
