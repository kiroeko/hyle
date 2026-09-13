if(HYLE_COMPUTE_CUDA_ENABLED)
  # Keep Toolkit ownership in the CUDA factor so source and installed
  # consumers see the same public header contract while the Driver API remains
  # implementation-private.
  hyle_add_module(compute_backend_cuda SOURCES
    cuda_backend.cc
    cuda_memory.cc)
  target_include_directories(hyle_compute_backend_cuda PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute")
  target_link_libraries(hyle_compute_backend_cuda
    PUBLIC CUDA::cudart
    PRIVATE CUDA::cuda_driver)
  if(WIN32 AND
     (MSVC OR CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC"))
    # CUDA 13.3's Windows cuda.lib is a static driver-loader archive whose
    # objects inject /DEFAULTLIB:LIBCMT even into an otherwise coherent /MD
    # graph. HYLE and CUDART both use the dynamic CRT; resolve those loader
    # references there and reject only the conflicting vendor directive. A
    # static HYLE SDK must pass the same normalization to its final link.
    target_link_options(hyle_compute_backend_cuda PRIVATE
      /NODEFAULTLIB:LIBCMT)
    if(HYLE_STATIC)
      target_link_options(hyle_compute_backend_cuda INTERFACE
        /NODEFAULTLIB:LIBCMT)
    endif()
  endif()
  target_compile_definitions(hyle_compute_backend_cuda
    PUBLIC HYLE_COMPUTE_CUDA=1)
  hyle_register_package_dependency(
    NAME compute_cuda_sdk
    OWNER compute_backend_cuda
    FEATURES compute_cuda
    FIND_DEPENDENCIES CUDAToolkit|13.3)
else()
  hyle_add_module(compute_backend_cuda OPTIONAL_ABSENT)
endif()
