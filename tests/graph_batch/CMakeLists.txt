hyle_add_header_self_contained_test(graph_batch HEADER_DIR graph HEADERS batch.h detail/paged_append_source.h)
hyle_add_test(graph_batch SOURCES materialized_graph_test.cc native_paged_node_test.cc)
if(_HYLE_COMPUTE_HAVE_CPU AND HYLE_GGML_CPU_ENABLED AND
   TARGET hyle_compute_backend_cpu AND TARGET hyle_compute_provider_ggml)
  hyle_add_test(graph_batch NAME typed_capability
    SOURCES typed_capability_graph_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_provider_ggml hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_typed_capability_test)
  hyle_add_test(graph_batch NAME cpu_append SOURCES paged_append_graph_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_provider_ggml hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_cpu_append_test)
  hyle_add_test(graph_batch NAME inference_append SOURCES ../tools/inference_append_transactions_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_provider_ggml)
  hyle_stage_runtime(hyle_graph_batch_inference_append_test)
  hyle_add_test(graph_batch NAME inference_paged_sequence SOURCES ../tools/inference_paged_sequence_test.cc
    DEPS hyle_compute_backend_cpu hyle_compute_provider_ggml)
  hyle_stage_runtime(hyle_graph_batch_inference_paged_sequence_test)
  if(TARGET hyle_graph_route_cpu)
    hyle_add_test(graph_batch NAME cpu_encoded_pages
      SOURCES ../compute_batch/encoded_page_common_consumer.cc
      DEPS hyle_graph_route_cpu hyle_compute_provider_ggml hyle_simd)
    hyle_stage_runtime(hyle_graph_batch_cpu_encoded_pages_test)
  endif()
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND _HYLE_COMPUTE_HAVE_CUDA AND HYLE_GGML_CPU_ENABLED AND
   TARGET hyle_graph_route_cpu_cuda AND TARGET hyle_compute_route_cpu AND
   TARGET hyle_graph_route_cuda AND TARGET hyle_compute_provider_ggml AND
   TARGET hyle_compute_provider_flashinfer)
  hyle_add_test(graph_batch NAME cuda_encoded_pages
    SOURCES ../compute_batch/encoded_page_common_consumer.cc
    DEPS hyle_graph_route_cpu_cuda hyle_compute_route_cpu hyle_graph_route_cuda
      hyle_compute_provider_ggml hyle_compute_provider_flashinfer hyle_simd)
  target_compile_definitions(hyle_graph_batch_cuda_encoded_pages_test PRIVATE HYLE_ENCODED_CUDA=1)
  hyle_stage_runtime(hyle_graph_batch_cuda_encoded_pages_test)
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND _HYLE_COMPUTE_HAVE_VULKAN AND
   TARGET hyle_graph_route_cpu AND TARGET hyle_graph_route_vulkan AND
   TARGET hyle_graph_progress)
  hyle_add_test(graph_batch NAME vulkan_native_ready
    SOURCES native_paged_joint_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_vulkan hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_batch_vulkan_native_ready_test)
  hyle_add_test(graph_batch NAME vulkan_native_append
    SOURCES native_paged_append_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_vulkan hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_batch_vulkan_native_append_test)
  hyle_add_test(graph_batch NAME vulkan_native_capacity
    SOURCES native_paged_capacity_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_vulkan hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_batch_vulkan_native_capacity_test)
  hyle_add_test(graph_batch NAME vulkan_native_pending
    SOURCES ../compute_batch/pending_paged_receiver_native_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_vulkan hyle_graph_progress)
  target_compile_definitions(hyle_graph_batch_vulkan_native_pending_test PRIVATE HYLE_PENDING_NATIVE_JOINT_GRAPH=1)
  hyle_stage_runtime(hyle_graph_batch_vulkan_native_pending_test)
  hyle_add_test(graph_batch NAME paged_semantics
    SOURCES paged_semantics_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_vulkan hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_batch_paged_semantics_test)
  hyle_add_test(graph_batch NAME vulkan_encoded_pages
    SOURCES ../compute_batch/encoded_page_native_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_vulkan hyle_graph_progress hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_vulkan_encoded_pages_test)
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND _HYLE_COMPUTE_HAVE_APPLE AND
   TARGET hyle_graph_route_cpu AND TARGET hyle_graph_route_apple AND TARGET hyle_graph_progress)
  hyle_add_test(graph_batch NAME apple_native_ready
    SOURCES native_paged_joint_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_apple hyle_graph_progress)
  target_compile_definitions(hyle_graph_batch_apple_native_ready_test PRIVATE HYLE_JOINT_PAGE_APPLE=1)
  hyle_stage_runtime(hyle_graph_batch_apple_native_ready_test)
  hyle_add_test(graph_batch NAME apple_native_append
    SOURCES native_paged_append_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_apple hyle_graph_progress)
  target_compile_definitions(hyle_graph_batch_apple_native_append_test PRIVATE HYLE_JOINT_PAGE_APPLE=1)
  hyle_stage_runtime(hyle_graph_batch_apple_native_append_test)
  hyle_add_test(graph_batch NAME apple_native_capacity
    SOURCES native_paged_capacity_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_apple hyle_graph_progress)
  target_compile_definitions(hyle_graph_batch_apple_native_capacity_test PRIVATE HYLE_JOINT_PAGE_APPLE=1)
  hyle_stage_runtime(hyle_graph_batch_apple_native_capacity_test)
  hyle_add_test(graph_batch NAME apple_native_pending
    SOURCES ../compute_batch/pending_paged_receiver_native_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_apple hyle_graph_progress)
  target_compile_definitions(hyle_graph_batch_apple_native_pending_test PRIVATE
    HYLE_PENDING_NATIVE_JOINT_GRAPH=1 HYLE_PENDING_RECEIVER_APPLE=1)
  hyle_stage_runtime(hyle_graph_batch_apple_native_pending_test)
  hyle_add_test(graph_batch NAME apple_encoded_pages
    SOURCES ../compute_batch/encoded_page_native_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_apple hyle_graph_progress hyle_simd)
  target_compile_definitions(hyle_graph_batch_apple_encoded_pages_test PRIVATE HYLE_ENCODED_APPLE=1)
  hyle_stage_runtime(hyle_graph_batch_apple_encoded_pages_test)
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND TARGET hyle_graph_route_cpu)
  hyle_add_test(graph_batch NAME cpu_row_cache SOURCES row_cache_graph_test.cc row_cache_pending_graph_test.cc DEPS hyle_graph_route_cpu)
  hyle_add_test(graph_batch NAME cpu_materialized SOURCES materialized_controller_test.cc
    DEPS hyle_graph_route_cpu hyle_simd)
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND TARGET hyle_graph_route_cpu AND TARGET hyle_compute_provider_native)
  get_target_property(_row_cache_native_capabilities hyle_compute_provider_native HYLE_COMPUTE_CAPABILITIES)
  if("torch_cpu" IN_LIST _row_cache_native_capabilities)
    hyle_add_test(graph_batch NAME cpu_row_cache_indexer SOURCES row_cache_indexer_graph_test.cc
      DEPS hyle_graph_route_cpu hyle_compute_provider_native)
    hyle_stage_runtime(hyle_graph_batch_cpu_row_cache_indexer_test)
  endif()
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND _HYLE_COMPUTE_HAVE_APPLE AND
   TARGET hyle_graph_route_cpu AND TARGET hyle_graph_route_apple AND TARGET hyle_graph_progress)
  hyle_add_test(graph_batch NAME apple_public_mla
    SOURCES ../compute_batch/public_mla_native_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_apple hyle_graph_progress hyle_compute_provider_native hyle_simd)
  target_compile_definitions(hyle_graph_batch_apple_public_mla_test PRIVATE HYLE_TEST_MLA_APPLE=1)
  hyle_stage_runtime(hyle_graph_batch_apple_public_mla_test)
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND _HYLE_COMPUTE_HAVE_VULKAN AND
   TARGET hyle_graph_route_cpu AND TARGET hyle_graph_route_vulkan AND TARGET hyle_graph_progress)
  hyle_add_test(graph_batch NAME vulkan_public_mla
    SOURCES ../compute_batch/public_mla_native_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_route_vulkan hyle_graph_progress hyle_compute_provider_native hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_vulkan_public_mla_test)
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND HYLE_GGML_CPU_ENABLED AND TARGET hyle_compute_route_cpu)
  hyle_add_test(graph_batch NAME cpu_public_mla
    SOURCES ../compute_batch/public_mla_cpu_consumer.cc
    DEPS hyle_compute_route_cpu hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_cpu_public_mla_test)
endif()
foreach(_page_route IN ITEMS apple vulkan)
  string(TOUPPER "${_page_route}" _page_route_upper)
  if(_HYLE_COMPUTE_HAVE_CPU AND _HYLE_COMPUTE_HAVE_${_page_route_upper} AND
     TARGET hyle_graph_route_cpu AND TARGET hyle_graph_route_${_page_route} AND TARGET hyle_graph_progress)
    foreach(_page_case IN ITEMS ready pending producers)
      if(_page_case STREQUAL "ready")
        set(_page_source native_paged_joint_risks.cc)
      else()
        set(_page_source native_pending_joint_risks.cc)
      endif()
      set(_page_test hyle_graph_batch_${_page_route}_${_page_case}_risks_test)
      hyle_add_test(graph_batch NAME ${_page_route}_${_page_case}_risks
        SOURCES ${_page_source}
        DEPS hyle_graph_route_cpu hyle_graph_route_${_page_route} hyle_graph_progress hyle_simd)
      if(_page_route STREQUAL "apple")
        target_compile_definitions(${_page_test} PRIVATE HYLE_JOINT_PAGE_APPLE=1)
      else()
        target_include_directories(${_page_test} PRIVATE "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
      endif()
      if(_page_case STREQUAL "producers")
        target_compile_definitions(${_page_test} PRIVATE HYLE_PENDING_MULTI_NATIVE_PRODUCERS=1)
      endif()
      hyle_stage_runtime(${_page_test})
    endforeach()
  endif()
endforeach()
unset(_page_route)
unset(_page_route_upper)
unset(_page_case)
unset(_page_source)
unset(_page_test)
if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_graph_route_cuda)
  hyle_add_test(graph_batch NAME cuda_append_source SOURCES paged_append_source_test.cc
    DEPS hyle_graph_route_cuda hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_batch_cuda_append_source_test)
  hyle_add_test(graph_batch NAME cuda_row_cache_indexer SOURCES row_cache_indexer_graph_test.cc DEPS hyle_graph_route_cuda)
  target_compile_definitions(hyle_graph_batch_cuda_row_cache_indexer_test PRIVATE HYLE_ROW_CACHE_GRAPH_CUDA=1)
  hyle_stage_runtime(hyle_graph_batch_cuda_row_cache_indexer_test)
  hyle_add_test(graph_batch NAME cuda_row_cache SOURCES row_cache_graph_test.cc row_cache_pending_graph_test.cc DEPS hyle_graph_route_cuda)
  target_compile_definitions(hyle_graph_batch_cuda_row_cache_test PRIVATE HYLE_ROW_CACHE_GRAPH_CUDA=1)
  hyle_stage_runtime(hyle_graph_batch_cuda_row_cache_test)
endif()
if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_route_cuda AND
   TARGET hyle_compute_provider_flashinfer)
  if(TARGET hyle_graph_route_cuda)
    hyle_add_test(graph_batch NAME cuda_capture SOURCES batch_capture_test.cc
      DEPS hyle_graph_route_cuda hyle_simd)
    hyle_stage_runtime(hyle_graph_batch_cuda_capture_test)
    hyle_add_test(graph_batch NAME cuda_append SOURCES paged_append_cuda_consumer.cc
      DEPS hyle_graph_route_cuda hyle_simd)
    hyle_stage_runtime(hyle_graph_batch_cuda_append_test)
  endif()
  hyle_add_test(graph_batch NAME cuda_public_native
    SOURCES ../compute_batch/public_batch_cuda_consumer.cc
    DEPS hyle_compute_route_cuda hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_cuda_public_native_test)
  hyle_add_test(graph_batch NAME cuda_public_mla
    SOURCES ../compute_batch/public_mla_cuda_consumer.cc
    DEPS hyle_compute_route_cuda hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_cuda_public_mla_test)
  hyle_add_test(graph_batch NAME cuda_public_mla_pending
    SOURCES ../compute_batch/public_mla_pending_cuda_consumer.cc
    DEPS hyle_compute_route_cuda hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_cuda_public_mla_pending_test)
  hyle_add_test(graph_batch NAME cuda_public_mla_pending_graph
    SOURCES ../compute_batch/public_mla_pending_graph_cuda_consumer.cc
    DEPS hyle_compute_route_cuda hyle_simd)
  hyle_stage_runtime(hyle_graph_batch_cuda_public_mla_pending_graph_test)
endif()
