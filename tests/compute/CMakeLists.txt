hyle_add_header_self_contained_test(compute)
hyle_add_test(compute SOURCES completion_test.cc shape_domain_test.cc tensor_test.cc)
target_include_directories(hyle_compute_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute")
