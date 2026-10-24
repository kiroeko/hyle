hyle_add_header_self_contained_test(graph_route_vulkan
  HEADER_DIR graph HEADERS vulkan.h)
if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_test(graph_route_vulkan NAME float8_quantize_output
    SOURCES ../compute_models/float8_quantize_output_test.cc
    DEFINES HYLE_TEST_FP8_QUANT_VULKAN=1)
  # The byte-transfer fixture imports the route's original native storage.
  target_include_directories(hyle_graph_route_vulkan_float8_quantize_output_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
  hyle_stage_runtime(hyle_graph_route_vulkan_float8_quantize_output_test)
  hyle_add_test(graph_route_vulkan NAME ragged_attention
    SOURCES ../graph_batch/native_ragged_attention_test.cc
    DEPS hyle_graph_batch hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_ragged_attention_test)
  hyle_add_test(graph_route_vulkan NAME paged_row_indexer
    SOURCES ../graph_batch/native_row_cache_indexer_test.cc
    DEFINES HYLE_NATIVE_ROW_INDEXER_VULKAN=1 DEPS hyle_graph_batch)
  hyle_stage_runtime(hyle_graph_route_vulkan_paged_row_indexer_test)
  hyle_add_test(graph_route_vulkan NAME row_projection
    SOURCES row_cache_projection_test.cc row_cache_pending_projection_test.cc
    DEPS hyle_graph_batch)
  target_include_directories(hyle_graph_route_vulkan_row_projection_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/compute_ggml_runtime")
  hyle_stage_runtime(hyle_graph_route_vulkan_row_projection_test)
  add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/affine.spv"
    COMMAND "${HYLE_VULKAN_GLSLANG}" -V --target-env vulkan1.2 -S comp
      -o "${CMAKE_CURRENT_BINARY_DIR}/affine.spv"
      "${HYLE_SOURCE_DIR}/tests/compute_route_vulkan/affine.comp"
    DEPENDS "${HYLE_SOURCE_DIR}/tests/compute_route_vulkan/affine.comp" VERBATIM)
  hyle_add_test(graph_route_vulkan SOURCES graph_test.cc lease_binding_test.cc tensor_test.cc view_test.cc gate_norm_test.cc concat_test.cc reduction_test.cc managed_region_test.cc d5_qwen35_execution_test.cc
    topk_test.cc gather_test.cc scan_test.cc scatter_test.cc recurrent_test.cc execution_source_test.cc native_region_test.cc weight_binding_test.cc expert_working_set_test.cc
    mxfp4_test.cc gguf_test.cc iree_test.cc mature_tensor_test.cc catalogue_test.cc batched_matmul_test.cc nvfp4_reference_test.cc grouped_nvfp4_reference_test.cc local_nvfp4_reference_test.cc mxfp4_reference_test.cc blockwise_w4a8_test.cc fp8_channel_token_qdq_test.cc ct_block128_qdq_test.cc ct_block128_activation_edges_test.cc grouped_fp8_power2_test.cc e8m0_weight_import_test.cc mlx_orientation_test.cc mlx_head_payload_test.cc compact_mla_e4_test.cc paged_e4_kv_test.cc gptq_reference_test.cc static_fp8_reference_test.cc ct_packed_linear_test.cc ct_packed_grouped_test.cc "${CMAKE_CURRENT_BINARY_DIR}/affine.spv")
  if(TARGET hyle_graph_route_cpu)
    target_link_libraries(hyle_graph_route_vulkan_test PRIVATE hyle_graph_route_cpu)
    target_compile_definitions(hyle_graph_route_vulkan_test PRIVATE
      HYLE_TEST_VULKAN_CPU_CONSTANT=1)
  endif()
  target_compile_definitions(hyle_graph_route_vulkan_test PRIVATE
    HYLE_VULKAN_TEST_SHADER="${CMAKE_CURRENT_BINARY_DIR}/affine.spv")
  target_link_libraries(hyle_graph_route_vulkan_test PRIVATE
    hyle_compute_provider_ggml_vulkan hyle_compute_ggml_runtime ggml hyle_graph_progress hyle_graph_batch)
  target_include_directories(hyle_graph_route_vulkan_test SYSTEM PRIVATE
    "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_add_test(graph_route_vulkan NAME finite_native
    SOURCES ../graph/route_symmetry_vulkan.cc)
  hyle_stage_runtime(hyle_graph_route_vulkan_finite_native_test)
  hyle_add_test(graph_route_vulkan NAME retained_output_reuse
    SOURCES retained_output_reuse_consumer.cc)
  hyle_stage_runtime(hyle_graph_route_vulkan_retained_output_reuse_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN)

hyle_add_test(graph_route_vulkan NAME sampling_consumer
  SOURCES ../compute_route_vulkan/sampling_consumer.cc)
target_compile_definitions(hyle_graph_route_vulkan_sampling_consumer_test PRIVATE HYLE_G08_GRAPH=1)
hyle_stage_runtime(hyle_graph_route_vulkan_sampling_consumer_test)
if(_HYLE_COMPUTE_REQUEST_CPU AND TARGET hyle_graph_route_cpu AND TARGET hyle_graph_progress)
  hyle_add_test(graph_route_vulkan NAME native_effects
    SOURCES ../graph/route_effects_test.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress hyle_compute_backend_cpu
      hyle_compute_provider_native hyle_compute_provider_ggml)
  target_include_directories(hyle_graph_route_vulkan_native_effects_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/graph" "${HYLE_SOURCE_DIR}/src/compute")
  set_tests_properties(hyle_graph_route_vulkan_native_effects_test PROPERTIES
    ENVIRONMENT "GTEST_FILTER=FoundationEffects.*")
  hyle_stage_runtime(hyle_graph_route_vulkan_native_effects_test)
  hyle_add_test(graph_route_vulkan NAME public_effects
    SOURCES ../graph/public_effect_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_vulkan_public_effects_test)
  hyle_add_test(graph_route_vulkan NAME mixed_contracts
    SOURCES ../graph/cpu_vulkan_contracts.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_vulkan_mixed_contracts_test)
  hyle_add_test(graph_route_vulkan NAME mixed_public_consumer
    SOURCES ../../samples/package/graph/cpu_vulkan/main.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_vulkan_mixed_public_consumer_test)
  hyle_add_test(graph_route_vulkan NAME mixed_finite
    SOURCES ../graph/route_symmetry_vulkan.cc
    DEFINES HYLE_FOUNDATION_A_HYBRID=1
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_vulkan_mixed_finite_test)
  hyle_add_test(graph_route_vulkan NAME completion_publication
    SOURCES ../graph/cpu_completion_publication.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_vulkan_completion_publication_test)
  hyle_add_test(graph_route_vulkan NAME mixed_pipeline
    SOURCES ../graph/native_joint_pipeline.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_vulkan_mixed_pipeline_test)
  set_tests_properties(hyle_graph_route_vulkan_mixed_pipeline_test PROPERTIES TIMEOUT 600)
  hyle_add_test(graph_route_vulkan NAME mixed_sampling
    SOURCES ../graph_route_cpu_vulkan/sampling_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_vulkan_mixed_sampling_test)
endif()
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_vulkan NAME ga_indexer
    SOURCES ga_indexer_state_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_ga_indexer_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_vulkan NAME ga_ratio4
    SOURCES ga_ratio4_producer_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_ga_ratio4_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_vulkan NAME ga_mhc
    SOURCES ga_mhc_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_ga_mhc_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_test(graph_route_vulkan NAME ga_encoded_bank
    SOURCES ga_encoded_bank_test.cc DEPS hyle_graph_batch hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_ga_encoded_bank_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_vulkan NAME ga_full_ffn
    SOURCES ga_full_ffn_test.cc DEPS hyle_graph_batch hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_ga_full_ffn_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_vulkan NAME glm_flash_mixed_mlx
    SOURCES glm_flash_mixed_mlx_test.cc DEPS hyle_graph_batch hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_glm_flash_mixed_mlx_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_test(graph_route_vulkan NAME fp8_outer_scale
    SOURCES fp8_outer_scale_test.cc DEPS hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_fp8_outer_scale_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_test(graph_route_vulkan NAME fp8_scale_policy
    SOURCES fp8_scale_policy_test.cc DEPS hyle_simd)
  hyle_stage_runtime(hyle_graph_route_vulkan_fp8_scale_policy_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_test(graph_route_vulkan NAME bfloat_linear_consumer
    SOURCES bfloat_linear_consumer.cc)
  hyle_stage_runtime(hyle_graph_route_vulkan_bfloat_linear_consumer_test)
endif()

if(_HYLE_COMPUTE_HAVE_VULKAN)
  hyle_add_test(graph_route_vulkan NAME retained_attention_consumer
    SOURCES retained_attention_consumer.cc DEPS hyle_graph_batch)
  hyle_stage_runtime(hyle_graph_route_vulkan_retained_attention_consumer_test)
endif()
