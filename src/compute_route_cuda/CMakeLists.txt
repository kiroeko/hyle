if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_module(compute_route_cuda SOURCES cuda_context.cc)
else()
  hyle_add_module(compute_route_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_route_cuda PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
