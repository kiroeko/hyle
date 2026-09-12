hyle_add_header_self_contained_test(graph_provider_onednn
  HEADER_DIR graph/providers
  HEADERS onednn.h)

if(HYLE_COMPUTE_ONEDNN_ENABLED AND TARGET hyle_graph AND
   TARGET hyle_compute_backend_cpu AND TARGET hyle_compute_provider_native)
  hyle_add_test(graph_provider_onednn SOURCES onednn_region_provider_test.cc
    DEPS hyle_graph hyle_compute_backend_cpu hyle_compute_provider_native
         hyle_compute_provider_onednn)
endif()
