hyle_add_header_self_contained_test(util_runtime
  HEADER_DIR util
  HEADERS demangle.h registry.h)
hyle_add_test(util_runtime SOURCES registry_test.cc demangle_test.cc)
hyle_add_test(util_runtime NAME demangle_fault SOURCES demangle_fault_test.cc)
