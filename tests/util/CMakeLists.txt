hyle_add_header_self_contained_test(util)
hyle_add_test(util SOURCES ref_counted_test.cc scope_guard_test.cc type_name_test.cc registry_test.cc random_test.cc spin_lock_test.cc demangle_test.cc c_util_concurrency_test.cc DEPS hyle_container)
hyle_add_c_test(util c_util_test.c)
