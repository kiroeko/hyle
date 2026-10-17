hyle_ensure_module_test_target(graph_route_cpu_apple)
if(_HYLE_COMPUTE_HAVE_APPLE AND _HYLE_COMPUTE_REQUEST_CPU)
  hyle_add_header_self_contained_test(graph_route_cpu_apple
    HEADER_DIR graph HEADERS cpu_apple.h)
  hyle_add_test(graph_route_cpu_apple NAME contracts
    SOURCES ../graph/cpu_apple_contracts.cc DEPS hyle_graph_route_cpu)
  hyle_add_test(graph_route_cpu_apple NAME public_consumer
    SOURCES ../../samples/package/graph/cpu_apple/main.cc DEPS hyle_graph_route_cpu)
  hyle_add_test(graph_route_cpu_apple NAME finite
    SOURCES ../graph/route_symmetry_apple.cc DEPS hyle_graph_route_cpu)
  hyle_add_test(graph_route_cpu_apple NAME packed_weights
    SOURCES packed_weight_test.cc DEPS hyle_graph_route_cpu)
  hyle_add_test(graph_route_cpu_apple NAME completion_publication
    SOURCES ../graph/cpu_completion_publication.cc
    DEFINES HYLE_COMPLETION_PUBLICATION_APPLE=1 DEPS hyle_graph_route_cpu)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_contracts_test)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_public_consumer_test)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_finite_test)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_packed_weights_test)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_completion_publication_test)
endif()

if(_HYLE_COMPUTE_HAVE_APPLE AND _HYLE_COMPUTE_REQUEST_CPU)
  hyle_add_test(graph_route_cpu_apple NAME sampling_consumer
    SOURCES sampling_consumer.cc DEPS hyle_graph_route_cpu)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_sampling_consumer_test)
endif()
