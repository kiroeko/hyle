hyle_add_header_self_contained_test(util)
hyle_add_test(util SOURCES registry_test.cc demangle_test.cc c_util_concurrency_test.cc)
hyle_add_c_test(util c_util_test.c)
hyle_add_c_surface_test(util)
