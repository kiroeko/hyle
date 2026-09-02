hyle_add_header_self_contained_test(timer_queue
  HEADER_DIR time
  HEADERS common_timeout_queue.h timer_entry.h timer_queue.h)
hyle_add_test(timer_queue SOURCES header_test.cc timer_queue_test.cc
  common_timeout_queue_test.cc macro_composition_test.cc)
