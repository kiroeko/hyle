if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_module(graph_route_cuda SOURCES cuda_context.cc)
  target_include_directories(hyle_graph_route_cuda PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_route_cuda")
else()
  hyle_add_module(graph_route_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_graph_route_cuda PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
