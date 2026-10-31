hyle_add_header_self_contained_test(compute_state HEADER_DIR compute
  HEADERS state_versions.h state_clone.h detail/state_clone_access.h)
hyle_add_test(compute_state SOURCES state_versions_test.cc state_clone_test.cc DEPS hyle_compute_backend_cpu)
hyle_add_test(compute_state NAME dspark_block5_policy
  SOURCES ../compute_models/dspark_block5_policy_test.cc)

# Small native-copy lifetime consumer; it has its own main and no model math.
# Reuse only routes already selected by the product, without enabling others.
set(_state_native_copy_dependencies "")
if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_backend_cuda)
  list(APPEND _state_native_copy_dependencies hyle_compute_backend_cuda)
endif()
if(_HYLE_COMPUTE_HAVE_APPLE AND TARGET hyle_compute_route_apple)
  list(APPEND _state_native_copy_dependencies hyle_compute_route_apple)
endif()
if(_HYLE_COMPUTE_HAVE_VULKAN AND TARGET hyle_compute_route_vulkan)
  list(APPEND _state_native_copy_dependencies hyle_compute_route_vulkan)
endif()
if(_state_native_copy_dependencies)
  add_executable(hyle_compute_state_native_fault_consumer native_fault_consumer.cc)
  set_target_properties(hyle_compute_state_native_fault_consumer PROPERTIES
    HYLE_TEST_MODULE compute_state)
  target_compile_features(hyle_compute_state_native_fault_consumer PRIVATE cxx_std_23)
  target_link_libraries(hyle_compute_state_native_fault_consumer PRIVATE
    hyle_compute_state ${_state_native_copy_dependencies})
  if(_HYLE_COMPUTE_HAVE_VULKAN)
    # The existing unknown-fence observation boundary uses the native dispatch
    # header. This is test-only and does not add SDK types to a public header.
    target_include_directories(hyle_compute_state_native_fault_consumer SYSTEM PRIVATE
      "$<TARGET_PROPERTY:Vulkan-Headers,INTERFACE_INCLUDE_DIRECTORIES>")
  endif()
  hyle_apply_warnings(hyle_compute_state_native_fault_consumer)
  hyle_apply_sanitizer(hyle_compute_state_native_fault_consumer)
  hyle_stage_runtime(hyle_compute_state_native_fault_consumer)
  if(TARGET hyle_compute_state_tests)
    add_dependencies(hyle_compute_state_tests hyle_compute_state_native_fault_consumer)
  endif()
  add_test(NAME hyle_compute_state_native_fault_consumer
    COMMAND hyle_compute_state_native_fault_consumer)
  set_tests_properties(hyle_compute_state_native_fault_consumer PROPERTIES LABELS compute_state)
endif()
unset(_state_native_copy_dependencies)
if(_HYLE_COMPUTE_HAVE_CUDA AND HYLE_GGML_CUDA_ENABLED AND
   TARGET hyle_graph_route_cuda AND TARGET hyle_compute_provider_cuda)
  hyle_add_test(compute_state NAME row_cache_cuda
    SOURCES row_cache_consumer.cc DEPS hyle_graph_route_cuda hyle_simd)
  target_compile_definitions(hyle_compute_state_row_cache_cuda_test PRIVATE HYLE_ROW_CACHE_CUDA=1)
  hyle_stage_runtime(hyle_compute_state_row_cache_cuda_test)
endif()
if(_HYLE_COMPUTE_HAVE_CPU AND HYLE_GGML_CPU_ENABLED AND
   TARGET hyle_graph_route_cpu AND TARGET hyle_compute_provider_native)
  hyle_add_test(compute_state NAME row_cache
    SOURCES row_cache_consumer.cc DEPS hyle_graph_route_cpu hyle_simd)
  hyle_stage_runtime(hyle_compute_state_row_cache_test)
  get_target_property(_hyle_state_native_capabilities hyle_compute_provider_native
    HYLE_COMPUTE_CAPABILITIES)
  if("torch_cpu" IN_LIST _hyle_state_native_capabilities)
    hyle_add_test(compute_state NAME deepseek_sparse_prefix
      SOURCES deepseek_sparse_prefix_consumer.cc DEPS hyle_graph_route_cpu hyle_simd)
    hyle_stage_runtime(hyle_compute_state_deepseek_sparse_prefix_test)
    hyle_add_test(compute_state NAME dspark_prefix
      SOURCES dspark_prefix_consumer_test.cc DEPS hyle_graph_route_cpu hyle_simd)
    hyle_stage_runtime(hyle_compute_state_dspark_prefix_test)
  endif()
endif()
