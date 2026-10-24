if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_module(graph_route_cuda SOURCES cuda_context.cc weight_binding.cc
    dense_bfloat_scores_region.cc ct_packed_linear_region.cc fp8_shared_prologue_region.cc)
  target_include_directories(hyle_graph_route_cuda PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_route_cuda")
else()
  hyle_add_module(graph_route_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_graph_route_cuda PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
