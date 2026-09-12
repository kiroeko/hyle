if(HYLE_COMPUTE_ONEDNN_ENABLED)
  hyle_add_module(graph_provider_onednn SOURCES onednn_region_provider.cc)
  target_include_directories(hyle_graph_provider_onednn PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute_provider_onednn")
  target_compile_definitions(hyle_graph_provider_onednn
    PUBLIC HYLE_GRAPH_ONEDNN=1)
else()
  hyle_add_module(graph_provider_onednn OPTIONAL_ABSENT)
endif()
