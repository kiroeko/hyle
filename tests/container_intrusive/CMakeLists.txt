hyle_add_header_self_contained_test(container_intrusive
  HEADER_DIR container
  HEADERS intrusive_list.h
  NO_C)
hyle_add_test(container_intrusive SOURCES intrusive_list_test.cc)
