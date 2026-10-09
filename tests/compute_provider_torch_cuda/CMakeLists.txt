hyle_add_header_self_contained_test(compute_provider_torch_cuda
  HEADER_DIR compute/providers HEADERS torch_cuda.h)
if(HYLE_COMPUTE_CUDA_ENABLED)
  hyle_add_test(compute_provider_torch_cuda SOURCES failure_test.cc mlx_precision_test.cc
    DEPS hyle_compute_route_cuda hyle_graph_route_cuda hyle_compute_backend_cuda)
  get_target_property(_cuda_leaf hyle_vendor_c10_cuda IMPORTED_LOCATION)
  get_filename_component(_cuda_lib_directory "${_cuda_leaf}" DIRECTORY)
  target_include_directories(hyle_compute_provider_torch_cuda_test SYSTEM PRIVATE
    "${_cuda_lib_directory}/../include")
  target_link_libraries(hyle_compute_provider_torch_cuda_test PRIVATE
    torch torch_cpu c10 hyle_vendor_c10_cuda CUDA::cudart)
  if(TORCH_CXX_FLAGS)
    separate_arguments(_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(hyle_compute_provider_torch_cuda_test PRIVATE ${_torch_flags})
  endif()
  hyle_stage_runtime(hyle_compute_provider_torch_cuda_test)
endif()
