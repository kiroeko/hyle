hyle_add_header_self_contained_test(util_core
  HEADER_DIR util
  HEADERS
    random.h
    ref_counted.h
    scope_guard.h
    spin_lock.h
    type_name.h)
hyle_add_test(util_core
  SOURCES
    random_test.cc
    ref_counted_test.cc
    scope_guard_test.cc
    spin_lock_test.cc
    type_name_test.cc)
