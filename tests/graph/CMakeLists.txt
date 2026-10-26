find_package(Threads REQUIRED)

hyle_add_header_self_contained_test(graph)
hyle_add_test(graph SOURCES graph_test.cc lease_submission_test.cc native_analysis_test.cc effect_preflight_test.cc slot_bindings_test.cc)
if(TARGET hyle_graph_route_cpu AND TARGET hyle_compute_backend_cpu
   AND TARGET hyle_compute_provider_native)
  hyle_add_test(graph NAME slots SOURCES slots_execution_test.cc
    DEPS hyle_graph_route_cpu hyle_compute_backend_cpu hyle_compute_provider_native)
  hyle_stage_runtime(hyle_graph_slots_test)
endif()
if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_graph_route_cuda
   AND TARGET hyle_compute_backend_cuda)
  hyle_add_test(graph NAME slots_cuda SOURCES slots_cuda_execution_test.cc output_source_cuda_test.cc
    DEPS hyle_graph_route_cuda hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_slots_cuda_test)
endif()
if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_provider_cublas)
  hyle_add_test(graph NAME native_retirement
    SOURCES aggregate_native_retirement_test.cc
    DEPS hyle_compute_backend_cuda hyle_compute_provider_cublas)
  target_include_directories(hyle_graph_native_retirement_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/graph")
  hyle_stage_runtime(hyle_graph_native_retirement_test)
endif()
if(_HYLE_COMPUTE_REQUEST_CPU AND HYLE_GGML_CPU_ENABLED
   AND TARGET hyle_compute_backend_cpu AND TARGET hyle_compute_provider_native
   AND TARGET hyle_compute_provider_ggml)
  hyle_add_test(graph NAME executable SOURCES
    executable_test.cc
    alias_liveness_test.cc
    constant_artifact_cache_test.cc
    configuration_input_test.cc
    execution_source_test.cc
    custom_kernel_test.cc
    input_source_frontier_test.cc
    invocation_unbind_test.cc
    region_provider_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_provider_native
         hyle_compute_provider_ggml Threads::Threads)
  hyle_stage_runtime(hyle_graph_executable_test)
  target_include_directories(hyle_graph_executable_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute")
  # OPTIONAL_ABSENT keeps a placeholder target so the component graph remains
  # queryable when CUDA is OFF. Only compile the real CUDA test when the
  # Toolkit-backed implementation was actually enabled.
  if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_backend_cuda)
    target_sources(hyle_graph_executable_test PRIVATE cuda_graph_test.cc)
    target_link_libraries(hyle_graph_executable_test PRIVATE
      hyle_compute_backend_cuda)
  endif()
endif()
