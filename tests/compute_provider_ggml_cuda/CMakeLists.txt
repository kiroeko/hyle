hyle_add_header_self_contained_test(compute_provider_ggml_cuda
  HEADER_DIR compute/providers HEADERS ggml_cuda.h)
if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_test(compute_provider_ggml_cuda NAME gdn_output_scale
    SOURCES ../compute_provider_native/gdn_output_scale_test.cc DEPS hyle_compute_backend_cuda)
  target_compile_definitions(hyle_compute_provider_ggml_cuda_gdn_output_scale_test PRIVATE HYLE_GDN_SCALE_CUDA=1)
  target_link_libraries(hyle_compute_provider_ggml_cuda_gdn_output_scale_test PRIVATE ggml)
  hyle_stage_runtime(hyle_compute_provider_ggml_cuda_gdn_output_scale_test)
  hyle_add_test(compute_provider_ggml_cuda SOURCES provider_test.cc allocation_test.cc sink_attention_test.cc hadamard_test.cc tensor_test.cc indexed_gdn_test.cc iq1_test.cc bfloat_matvec_test.cc packed_matmul_bounds_test.cc packed_matmul_tail_test.cc packed_mmq_workspace_test.cc
    DEPS hyle_compute_backend_cuda)
  target_include_directories(hyle_compute_provider_ggml_cuda_test PRIVATE
    "${HYLE_GGML_SOURCE_DIR}/src" "${HYLE_SOURCE_DIR}/src/compute_provider_ggml_cuda")
  target_link_libraries(hyle_compute_provider_ggml_cuda_test PRIVATE ggml)
  if(HYLE_STATIC AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_definitions(hyle_compute_provider_ggml_cuda_test PRIVATE
      HYLE_GGML_CUDA_PROVIDER_FAULTS=1)
    target_link_options(hyle_compute_provider_ggml_cuda_test PRIVATE
      LINKER:--wrap=cudaStreamBeginCapture)
  endif()
  hyle_stage_runtime(hyle_compute_provider_ggml_cuda_test)
  # Only the attention consumer needs the ordinary/default Graph assemblies.
  hyle_add_test(compute_provider_ggml_cuda NAME attention SOURCES attention_test.cc
    DEPS hyle_compute_backend_cuda hyle_compute_route_cuda hyle_graph_route_cuda)
  hyle_stage_runtime(hyle_compute_provider_ggml_cuda_attention_test)
endif()
