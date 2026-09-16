if(HYLE_COMPUTE_CUDA_ENABLED)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 80 89-real)
  endif()
  enable_language(CUDA)
  hyle_add_module(compute_provider_cuda SOURCES cuda_provider.cc native_kernel.cu)
  set_target_properties(hyle_compute_provider_cuda PROPERTIES
    CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON CUDA_RUNTIME_LIBRARY Shared)
  target_include_directories(hyle_compute_provider_cuda PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute_backend_cuda")
  target_link_libraries(hyle_compute_provider_cuda PRIVATE CUDA::cudart)
  target_compile_definitions(hyle_compute_provider_cuda PUBLIC HYLE_COMPUTE_CUDA=1)
  if(MSVC)
    target_compile_options(hyle_compute_provider_cuda PRIVATE
      "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8,/wd4211>")
  endif()
else()
  hyle_add_module(compute_provider_cuda OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_cuda PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
