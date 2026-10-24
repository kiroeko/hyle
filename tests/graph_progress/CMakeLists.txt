find_package(Threads REQUIRED)

hyle_add_header_self_contained_test(graph_progress
  HEADER_DIR graph HEADERS progress.h)
if(TARGET hyle_compute_backend_cpu AND TARGET hyle_compute_provider_native)
  hyle_add_test(graph_progress SOURCES progress_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_provider_native Threads::Threads)
  hyle_stage_runtime(hyle_graph_progress_test)
  set_tests_properties(hyle_graph_progress_test PROPERTIES TIMEOUT 60)
endif()
