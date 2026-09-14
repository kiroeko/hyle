hyle_add_header_self_contained_test(compute_provider_native
  HEADER_DIR compute
  HEADERS cpu.h providers/native.h)
hyle_add_test(compute_provider_native SOURCES
  ../compute/operation_test.cc
  cast_test.cc
  elementwise_test.cc
  indexing_test.cc
  matmul_gated_test.cc
  normalization_test.cc
  reduction_test.cc
  transform_test.cc
  transformer_test.cc)
