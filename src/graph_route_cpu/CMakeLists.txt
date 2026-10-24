hyle_add_module(graph_route_cpu SOURCES cpu_context.cc compile.cc ct_packed_linear_region.cc)

if(HYLE_GGML_CPU_ENABLED)
  target_sources(hyle_graph_route_cpu PRIVATE ggml_region.cc)
  target_include_directories(hyle_graph_route_cpu PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute"
    "${HYLE_SOURCE_DIR}/src/compute_provider_ggml")
endif()

if(HYLE_STATIC AND NOT _HYLE_COMPUTE_HAVE_ONEDNN)
  get_target_property(_route_links hyle_graph_route_cpu
    INTERFACE_LINK_LIBRARIES)
  foreach(_absent IN ITEMS compute_provider_onednn graph_provider_onednn)
    string(REPLACE "$<LINK_ONLY:hyle_${_absent}>"
      "$<BUILD_INTERFACE:$<LINK_ONLY:hyle_${_absent}>>"
      _route_links "${_route_links}")
  endforeach()
  set_property(TARGET hyle_graph_route_cpu PROPERTY
    INTERFACE_LINK_LIBRARIES "${_route_links}")
  unset(_absent)
  unset(_route_links)
endif()
