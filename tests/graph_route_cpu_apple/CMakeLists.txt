hyle_ensure_module_test_target(graph_route_cpu_apple)
if(HYLE_COMPUTE_APPLE_ENABLED AND HYLE_COMPUTE_CPU)
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
  hyle_stage_runtime(hyle_graph_route_cpu_apple_contracts_test)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_public_consumer_test)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_finite_test)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_packed_weights_test)
endif()

if(HYLE_COMPUTE_APPLE_ENABLED AND HYLE_COMPUTE_CPU)
  hyle_add_test(graph_route_cpu_apple NAME sampling_consumer
    SOURCES sampling_consumer.cc DEPS hyle_graph_route_cpu)
  hyle_stage_runtime(hyle_graph_route_cpu_apple_sampling_consumer_test)
endif()
