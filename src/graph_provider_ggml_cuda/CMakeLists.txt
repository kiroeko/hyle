if(NOT HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_module(graph_provider_ggml_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_graph_provider_ggml_cuda PROPERTY HYLE_INTERNAL_ONLY_ABSENT TRUE)
  return()
endif()
hyle_add_module(graph_provider_ggml_cuda SOURCES ggml_cuda_region.cc)
target_compile_definitions(hyle_graph_provider_ggml_cuda PUBLIC HYLE_COMPUTE_GGML_CUDA=1)
target_include_directories(hyle_graph_provider_ggml_cuda PRIVATE
  "${HYLE_SOURCE_DIR}/src/compute"
  "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_cuda")
