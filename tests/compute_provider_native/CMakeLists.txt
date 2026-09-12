hyle_add_header_self_contained_test(compute_provider_native
  HEADER_DIR compute
  HEADERS cpu.h providers/native.h)
hyle_add_test(compute_provider_native SOURCES ../compute/operation_test.cc)
