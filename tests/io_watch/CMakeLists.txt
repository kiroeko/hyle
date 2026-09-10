hyle_add_header_self_contained_test(io_watch
  HEADER_DIR io
  HEADERS watch.h)
hyle_add_test(io_watch SOURCES watch_test.cc watch_contract_test.cc
  watch_tree_test.cc
  DEPS hyle_platform_fs hyle_sched hyle_time)
