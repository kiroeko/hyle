hyle_add_test(core SOURCES core_test.cc intrusive_list_test.cc)
# Compile the Native FFI schema with a C compiler and call every real core
# symbol. This is an ABI conformance fixture, not a source-level C product API.
hyle_add_c_test(core c_core_test.c)
hyle_add_c_surface_test(core)
hyle_add_header_self_contained_test(core)
