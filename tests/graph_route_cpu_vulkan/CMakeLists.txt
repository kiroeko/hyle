hyle_ensure_module_test_target(graph_route_cpu_vulkan)
if(_HYLE_COMPUTE_HAVE_VULKAN AND _HYLE_COMPUTE_REQUEST_CPU)
  hyle_add_header_self_contained_test(graph_route_cpu_vulkan
    HEADER_DIR graph HEADERS cpu_vulkan.h)
  hyle_add_test(graph_route_cpu_vulkan NAME contracts
    SOURCES ../graph/cpu_vulkan_contracts.cc DEPS hyle_graph_route_cpu)
  hyle_add_test(graph_route_cpu_vulkan NAME public_consumer
    SOURCES ../../samples/package/graph/cpu_vulkan/main.cc DEPS hyle_graph_route_cpu)
  hyle_add_test(graph_route_cpu_vulkan NAME finite
    SOURCES ../graph/route_symmetry_vulkan.cc
    DEFINES HYLE_FOUNDATION_A_HYBRID=1 DEPS hyle_graph_route_cpu)
  hyle_add_test(graph_route_cpu_vulkan NAME completion_publication
    SOURCES ../graph/cpu_completion_publication.cc DEPS hyle_graph_route_cpu)
  hyle_stage_runtime(hyle_graph_route_cpu_vulkan_contracts_test)
  hyle_stage_runtime(hyle_graph_route_cpu_vulkan_public_consumer_test)
  hyle_stage_runtime(hyle_graph_route_cpu_vulkan_finite_test)
  hyle_stage_runtime(hyle_graph_route_cpu_vulkan_completion_publication_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN AND _HYLE_COMPUTE_REQUEST_CPU)
  hyle_add_test(graph_route_cpu_vulkan NAME sampling_consumer
    SOURCES sampling_consumer.cc DEPS hyle_graph_route_cpu)
  hyle_stage_runtime(hyle_graph_route_cpu_vulkan_sampling_consumer_test)
endif()
