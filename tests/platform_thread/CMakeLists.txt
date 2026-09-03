hyle_add_header_self_contained_test(platform_thread
  HEADER_DIR platform
  HEADERS thread.h thread_convenience.h thread_visibility.h wait.h)
hyle_add_test(platform_thread SOURCES thread_test.cc wait_test.cc
  DEPS hyle_platform)
