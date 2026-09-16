hyle_add_module(graph_route_cpu SOURCES cpu_context.cc)

if(HYLE_STATIC AND NOT HYLE_COMPUTE_ONEDNN_ENABLED)
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
