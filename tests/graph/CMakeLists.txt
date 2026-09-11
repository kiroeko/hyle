find_package(Threads REQUIRED)

hyle_add_header_self_contained_test(graph)
hyle_add_test(graph SOURCES graph_test.cc executable_test.cc
  DEPS Threads::Threads)
