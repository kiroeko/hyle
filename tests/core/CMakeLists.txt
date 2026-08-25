hyle_add_test(core SOURCES core_test.cc intrusive_list_test.cc)
# close-01 sample: pure C TU (C compiler) linking only the core C ABI header.
hyle_add_c_test(core c_core_test.c)
hyle_add_c_surface_test(core)
hyle_add_header_self_contained_test(core CHECK_C_NODISCARD)
