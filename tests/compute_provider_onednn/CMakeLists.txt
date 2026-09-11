hyle_add_header_self_contained_test(compute_provider_onednn
  HEADER_DIR compute/providers
  HEADERS onednn.h)

if(HYLE_COMPUTE_ONEDNN_ENABLED)
  hyle_add_test(compute_provider_onednn SOURCES onednn_provider_test.cc)
  if(TARGET hyle_graph)
    hyle_add_test(compute_provider_onednn NAME graph
      SOURCES onednn_graph_test.cc
      DEPS hyle_graph)
  endif()
endif()
