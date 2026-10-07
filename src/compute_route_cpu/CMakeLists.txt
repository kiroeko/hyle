hyle_add_module(compute_route_cpu SOURCES cpu_context.cc default_policy.cc)

if(HYLE_STATIC AND NOT HYLE_COMPUTE_ONEDNN_ENABLED)
  # The absent provider keeps its source-graph identity but contributes no
  # symbols or exported target to an installed Native-only CPU route.
  get_target_property(_route_links hyle_compute_route_cpu
    INTERFACE_LINK_LIBRARIES)
  string(REPLACE "$<LINK_ONLY:hyle_compute_provider_onednn>"
    "$<BUILD_INTERFACE:$<LINK_ONLY:hyle_compute_provider_onednn>>"
    _route_links "${_route_links}")
  set_property(TARGET hyle_compute_route_cpu PROPERTY
    INTERFACE_LINK_LIBRARIES "${_route_links}")
  unset(_route_links)
endif()
