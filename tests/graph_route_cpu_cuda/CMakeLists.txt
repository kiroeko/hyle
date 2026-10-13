hyle_ensure_module_test_target(graph_route_cpu_cuda)
if(HYLE_COMPUTE_CUDA_ENABLED AND HYLE_COMPUTE_CPU)
  get_filename_component(_hyle_cuda_test_bin "${CMAKE_CUDA_COMPILER}" DIRECTORY)
  find_path(_hyle_cuda_test_include cuda.h REQUIRED
    HINTS "${_hyle_cuda_test_bin}/../include"
          "${_hyle_cuda_test_bin}/../targets/x86_64-linux/include")
  hyle_add_header_self_contained_test(graph_route_cpu_cuda
    HEADER_DIR graph HEADERS cpu_cuda.h EXTRA_INCLUDE_DIRS "${_hyle_cuda_test_include}")
  hyle_add_test(graph_route_cpu_cuda NAME contracts
    SOURCES ../graph/cpu_cuda_contracts.cc DEPS hyle_graph_route_cpu hyle_graph_route_cuda)
  hyle_add_test(graph_route_cpu_cuda NAME public_consumer
    SOURCES ../../samples/package/graph/cpu_cuda/main.cc DEPS hyle_graph_route_cpu hyle_graph_route_cuda)
  hyle_stage_runtime(hyle_graph_route_cpu_cuda_contracts_test)
  hyle_stage_runtime(hyle_graph_route_cpu_cuda_public_consumer_test)
endif()
