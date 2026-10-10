if(HYLE_COMPUTE_CUDA_ENABLED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80 89-real)
  endif()
  enable_language(CUDA)
  include("${CMAKE_CURRENT_SOURCE_DIR}/HyleCudaSdkConfig.cmake")
  # Keep Toolkit ownership in the CUDA factor so source and installed
  # consumers see the same public header contract while the Driver API remains
  # implementation-private.
  hyle_add_module(compute_backend_cuda TESTING SOURCES
    cuda_backend.cc
    driver_release.cc
    cuda_memory.cc
    cuda_device_status.cc
    device_status_kernel.cu)
  set_target_properties(hyle_compute_backend_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  if(NOT MSVC AND NOT HYLE_SANITIZER STREQUAL "off")
    target_compile_options(hyle_compute_backend_cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
    target_link_options(hyle_compute_backend_cuda PRIVATE
      "$<$<LINK_LANGUAGE:CUDA>:--forward-unknown-to-host-compiler>")
  endif()
  if(MSVC)
    target_compile_options(hyle_compute_backend_cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4211>")
  endif()
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
    FIND_DEPENDENCIES CUDAToolkit|13.3 HyleCudaSdk
    DEVELOPMENT_FILE_SPECS
      "${CMAKE_CURRENT_SOURCE_DIR}/HyleCudaSdkConfig.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleCudaSdk|HyleCudaSdkConfig.cmake")
else()
  hyle_add_module(compute_backend_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_backend_cuda PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
