if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_module(compute_provider_cublas SOURCES cublas_provider.cc)
  target_include_directories(hyle_compute_provider_cublas PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute"
    "${CMAKE_CURRENT_SOURCE_DIR}/../compute_backend_cuda")
  target_link_libraries(hyle_compute_provider_cublas PRIVATE CUDA::cublasLt)
  if(TARGET HyleCudaSdk::cublasLt_runtime)
    # A direct SHARED edge survives export as a private DLL dependency.
    target_link_libraries(hyle_compute_provider_cublas PRIVATE HyleCudaSdk::cublasLt_runtime)
  endif()
  hyle_target_compute_capability(hyle_compute_provider_cublas cuda)
  hyle_register_package_dependency(
    NAME compute_cublas_sdk
    OWNER compute_provider_cublas
    FIND_DEPENDENCIES CUDAToolkit|13.3)
else()
  hyle_add_module(compute_provider_cublas OPTIONAL_ABSENT)
  set_property(TARGET hyle_compute_provider_cublas PROPERTY
    HYLE_INTERNAL_ONLY_ABSENT TRUE)
endif()
