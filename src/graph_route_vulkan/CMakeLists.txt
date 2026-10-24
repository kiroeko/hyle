if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_module(graph_route_vulkan TRACE SOURCES
    vulkan_lowering.cc vulkan_batch.cc execution_source.cc native_region.cc
    native_paged_region.cc native_paged_append.cc native_ragged_region.cc
    native_paged_pending.cc weight_binding.cc)
else()
  hyle_add_module(graph_route_vulkan OPTIONAL_ABSENT)
  set_property(TARGET hyle_graph_route_vulkan PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
