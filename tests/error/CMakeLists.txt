hyle_add_test(error SOURCES status_test.cc)
# close-01 sample: pure C TU (C compiler) linking only the C ABI header.
hyle_add_c_test(error c_error_test.c)
hyle_add_c_surface_test(error)
hyle_add_header_self_contained_test(error)
