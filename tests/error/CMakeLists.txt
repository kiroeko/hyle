hyle_add_test(error SOURCES status_test.cc)
# C-compiler/schema/link fixture for the pre-release Native FFI candidate.
hyle_add_c_test(error c_error_test.c)
hyle_add_c_surface_test(error)
hyle_add_header_self_contained_test(error)
