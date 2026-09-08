hyle_add_test(io_transfer SOURCES transfer_test.cc
  DEPS hyle_sched hyle_io_http)
hyle_add_header_self_contained_test(io_transfer HEADER_DIR io HEADERS transfer.h)
