hyle_add_header_self_contained_test(compute_provider_cuda
  HEADER_DIR compute/providers HEADERS cuda.h)
if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_test(compute_provider_cuda NAME packed_consumer
    SOURCES packed_resident_consumer.cc DEPS hyle_compute_backend_cuda hyle_compute_route_cuda)
  hyle_stage_runtime(hyle_compute_provider_cuda_packed_consumer_test)
  hyle_add_test(compute_provider_cuda NAME row_cache SOURCES row_cache_test.cc row_cache_paged_test.cc row_cache_indexer_test.cc row_cache_padded_test.cc
    DEPS hyle_compute_backend_cuda hyle_compute_batch hyle_compute_route_cuda hyle_graph)
  hyle_add_test(compute_provider_cuda SOURCES cuda_provider_test.cc grouped_fp8_test.cc tensor_test.cc scatter_test.cc sglang_primitives_test.cc collectives_test.cc gather_collectives_test.cc
    DEPS hyle_compute_backend_cuda hyle_graph)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA)
  enable_language(CUDA)
  hyle_add_test(compute_provider_cuda NAME sampling_math
    SOURCES sampling_math_test.cc sampling_math_probe.cu
    DEPS hyle_compute_backend_cuda)
  set_target_properties(hyle_compute_provider_cuda_sampling_math_test PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(hyle_compute_provider_cuda_sampling_math_test PRIVATE CUDA::cudart)
  hyle_stage_runtime(hyle_compute_provider_cuda_sampling_math_test)
endif()
