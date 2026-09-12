find_package(Threads REQUIRED)

hyle_add_header_self_contained_test(graph)
hyle_add_test(graph SOURCES graph_test.cc)
if(TARGET hyle_compute_backend_cpu AND TARGET hyle_compute_provider_native)
  hyle_add_test(graph NAME executable SOURCES executable_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_provider_native Threads::Threads)
endif()
