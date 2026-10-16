hyle_add_header_self_contained_test(compute_provider_cuda
  HEADER_DIR compute/providers HEADERS cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_provider_cuda SOURCES cuda_provider_test.cc tensor_test.cc scatter_test.cc sglang_primitives_test.cc collectives_test.cc gather_collectives_test.cc
    DEPS hyle_compute_backend_cuda)
endif()

if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_provider_cuda NAME sampling_math
    SOURCES sampling_math_test.cc sampling_math_probe.cu
    DEPS hyle_compute_backend_cuda)
  set_target_properties(hyle_compute_provider_cuda_sampling_math_test PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  target_link_libraries(hyle_compute_provider_cuda_sampling_math_test PRIVATE CUDA::cudart)
  hyle_stage_runtime(hyle_compute_provider_cuda_sampling_math_test)
endif()
