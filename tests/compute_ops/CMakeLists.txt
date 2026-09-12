hyle_add_header_self_contained_test(compute_ops
  HEADER_DIR compute
  HEADERS ops_visibility.h tensor_desc.h operation.h)
hyle_add_test(compute_ops SOURCES operation_test.cc tensor_desc_test.cc)
