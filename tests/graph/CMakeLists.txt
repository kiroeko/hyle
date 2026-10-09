find_package(Threads REQUIRED)

hyle_add_header_self_contained_test(graph)
hyle_add_test(graph SOURCES graph_test.cc)
if(HYLE_COMPUTE_CPU AND HYLE_GGML_CPU_ENABLED
   AND TARGET hyle_compute_backend_cpu AND TARGET hyle_compute_provider_native
   AND TARGET hyle_compute_provider_ggml)
  hyle_add_test(graph NAME executable SOURCES
    executable_test.cc
    custom_kernel_test.cc
    region_provider_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_provider_native
         hyle_compute_provider_ggml Threads::Threads)
  hyle_stage_runtime(hyle_graph_executable_test)
  target_include_directories(hyle_graph_executable_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute")
  # OPTIONAL_ABSENT keeps a placeholder target so the component graph remains
  # queryable when CUDA is OFF. Only compile the real CUDA test when the
  # Toolkit-backed implementation was actually enabled.
  if(HYLE_COMPUTE_CUDA_ENABLED AND TARGET hyle_compute_backend_cuda)
    target_sources(hyle_graph_executable_test PRIVATE cuda_graph_test.cc)
    target_link_libraries(hyle_graph_executable_test PRIVATE
      hyle_compute_backend_cuda)
  endif()
endif()
