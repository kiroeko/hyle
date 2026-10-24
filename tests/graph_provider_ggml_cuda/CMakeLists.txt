hyle_add_header_self_contained_test(graph_provider_ggml_cuda
  HEADER_DIR graph/providers HEADERS ggml_cuda.h)
if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_test(graph_provider_ggml_cuda SOURCES region_test.cc attention_region_test.cc
    dense_projection_test.cc constant_expression_test.cc rms_cast_fusion_test.cc
    gdn_l2_fusion_test.cc gdn_typed_load_test.cc conv_typed_load_test.cc cast_stable_silu_test.cc
    typed_gating_silu_test.cc paired_projection_test.cc
    DEPS hyle_graph hyle_graph_route_cuda hyle_compute_backend_cuda hyle_compute_provider_ggml_cuda)
  target_link_libraries(hyle_graph_provider_ggml_cuda_test PRIVATE ggml)
  hyle_stage_runtime(hyle_graph_provider_ggml_cuda_test)
endif()
