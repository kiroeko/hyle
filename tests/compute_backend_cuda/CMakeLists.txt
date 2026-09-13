if(HYLE_COMPUTE_CUDA_ENABLED)
  # Product code is pure C++ and only discovers the machine CUDA SDK. This
  # consumer probe deliberately enables nvcc in its own scope to prove that a
  # normal .cu kernel can be submitted through the typed HYLE stream seam.
  enable_language(CUDA)
  hyle_add_header_self_contained_test(compute_backend_cuda
    HEADER_DIR compute
    HEADERS backends/cuda.h
    EXTRA_INCLUDE_DIRS ${HYLE_CUDA_PUBLIC_INCLUDE_DIRS})
  hyle_add_test(compute_backend_cuda SOURCES
    cuda_backend_test.cc
    cuda_test_kernel.cu)
  target_link_libraries(hyle_compute_backend_cuda_test PRIVATE
    CUDA::cuda_driver)
  set_target_properties(hyle_compute_backend_cuda_test PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES native)
  if(MSVC)
    # nvcc 13.3 feeds both NVIDIA's UTF-8 headers and its generated C stub to
    # cl. Normalize the source character set and suppress only the generated
    # nonstandard-extension diagnostic; ordinary HYLE C++ warnings stay intact.
    target_compile_options(hyle_compute_backend_cuda_test PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>"
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>")
    # The test calls Driver APIs directly to prove external-allocation owner
    # transfer, so it sees the same CUDA 13.3 cuda.lib directive normalized by
    # the shipping factor target.
    target_link_options(hyle_compute_backend_cuda_test PRIVATE
      /NODEFAULTLIB:LIBCMT)
  endif()
else()
  # OFF/AUTO-absent is a packaging state, not a fake runtime. Keep the module
  # registration complete without manufacturing a CUDA behavior test.
  hyle_ensure_module_test_target(compute_backend_cuda)
endif()
