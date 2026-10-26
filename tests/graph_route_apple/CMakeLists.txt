hyle_add_header_self_contained_test(graph_route_apple
  HEADER_DIR graph HEADERS apple.h)
if(_HYLE_COMPUTE_HAVE_APPLE)
  hyle_add_test(graph_route_apple NAME slot_store SOURCES slot_store_test.cc)
  hyle_stage_runtime(hyle_graph_route_apple_slot_store_test)
  hyle_add_test(graph_route_apple NAME float8_quantize_output
    SOURCES ../compute_models/float8_quantize_output_test.cc
    DEFINES HYLE_TEST_FP8_QUANT_APPLE=1)
  hyle_stage_runtime(hyle_graph_route_apple_float8_quantize_output_test)
  hyle_add_test(graph_route_apple NAME retained_output_budget
    SOURCES retained_output_budget_consumer.cc)
  hyle_stage_runtime(hyle_graph_route_apple_retained_output_budget_test)
  hyle_add_test(graph_route_apple NAME ragged_attention
    SOURCES ../graph_batch/native_ragged_attention_test.cc
    DEFINES HYLE_NATIVE_RAGGED_APPLE=1 DEPS hyle_graph_batch hyle_simd)
  hyle_stage_runtime(hyle_graph_route_apple_ragged_attention_test)
  hyle_add_test(graph_route_apple NAME paged_row_indexer
    SOURCES ../graph_batch/native_row_cache_indexer_test.cc
    DEFINES HYLE_NATIVE_ROW_INDEXER_APPLE=1 DEPS hyle_graph_batch)
  hyle_stage_runtime(hyle_graph_route_apple_paged_row_indexer_test)
  hyle_add_test(graph_route_apple NAME row_pending_projection
    SOURCES row_cache_pending_projection_test.cc DEPS hyle_graph_batch)
  hyle_stage_runtime(hyle_graph_route_apple_row_pending_projection_test)
  hyle_add_test(graph_route_apple NAME row_projection
    SOURCES row_cache_projection_test.cc DEPS hyle_graph_batch)
  hyle_stage_runtime(hyle_graph_route_apple_row_projection_test)
  hyle_add_test(graph_route_apple NAME attention_partials
    SOURCES attention_partials_test.cc DEPS hyle_graph_batch)
  hyle_stage_runtime(hyle_graph_route_apple_attention_partials_test)
  hyle_add_test(graph_route_apple NAME configured_source
    SOURCES configured_source_test.cc weight_binding_test.cc)
  hyle_stage_runtime(hyle_graph_route_apple_configured_source_test)
  hyle_add_test(graph_route_apple NAME ga_encoded_bank
    SOURCES ga_encoded_bank_test.cc)
  hyle_stage_runtime(hyle_graph_route_apple_ga_encoded_bank_test)
  hyle_add_test(graph_route_apple NAME ga_full_ffn
    SOURCES ga_full_ffn_test.cc)
  hyle_stage_runtime(hyle_graph_route_apple_ga_full_ffn_test)
  hyle_add_test(graph_route_apple NAME glm_flash_mixed_mlx
    SOURCES glm_flash_mixed_mlx_test.cc)
  hyle_stage_runtime(hyle_graph_route_apple_glm_flash_mixed_mlx_test)
  hyle_add_test(graph_route_apple NAME affine_metadata
    SOURCES affine_metadata_test.cc)
  hyle_stage_runtime(hyle_graph_route_apple_affine_metadata_test)
  hyle_add_test(graph_route_apple NAME fp8_outer_scale
    SOURCES fp8_outer_scale_test.cc)
  hyle_stage_runtime(hyle_graph_route_apple_fp8_outer_scale_test)
  hyle_add_test(graph_route_apple NAME fp8_scale_policy
    SOURCES fp8_scale_policy_test.cc)
  hyle_stage_runtime(hyle_graph_route_apple_fp8_scale_policy_test)
  if(TARGET hyle_graph_progress)
    hyle_add_test(graph_route_apple NAME cross_runtime
      SOURCES cross_runtime_test.cc DEPS hyle_graph_progress)
    hyle_stage_runtime(hyle_graph_route_apple_cross_runtime_test)
  endif()
  hyle_add_test(graph_route_apple SOURCES graph_test.cc lease_binding_test.cc tensor_test.cc topk_experts_test.cc reduction_test.cc indexing_test.cc scatter_test.cc recurrent_test.cc model_composition_test.cc d5_qwen35_execution_test.cc mlx_quantized_test.cc native_region_test.cc nvfp4_reference_test.cc grouped_nvfp4_reference_test.cc local_nvfp4_reference_test.cc mxfp4_reference_test.cc fp8_channel_token_qdq_test.cc gptq_reference_test.cc static_fp8_reference_test.cc ct_packed_linear_test.cc ct_packed_grouped_test.cc blockwise_w4a8_test.cc ct_block128_qdq_test.cc ct_block128_activation_edges_test.cc grouped_fp8_power2_test.cc e8m0_weight_import_test.cc compact_mla_e4_test.cc paged_e4_kv_test.cc expert_working_set_test.cc mlx_moe_test.cc mlx_orientation_test.cc mlx_head_payload_test.cc)
  target_link_libraries(hyle_graph_route_apple_test PRIVATE ggml hyle_graph_batch)
  target_include_directories(hyle_graph_route_apple_test PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_add_test(graph_route_apple NAME finite_native
    SOURCES ../graph/route_symmetry_apple.cc DEFINES HYLE_FOUNDATION_A_NATIVE_ONLY=1)
  hyle_stage_runtime(hyle_graph_route_apple_finite_native_test)
  hyle_add_test(graph_route_apple NAME packed_consumer
    SOURCES packed_resident_consumer.cc)
  hyle_stage_runtime(hyle_graph_route_apple_packed_consumer_test)
  if(_HYLE_COMPUTE_REQUEST_CPU)
    hyle_add_test(graph_route_apple NAME mixed_native
      SOURCES mixed_region_test.cc
      DEPS hyle_graph_route_cpu)
    hyle_stage_runtime(hyle_graph_route_apple_mixed_native_test)
    if(TARGET hyle_graph_route_cpu AND TARGET hyle_graph_progress)
      hyle_add_test(graph_route_apple NAME paged_consumer
        SOURCES paged_region_consumer.cc
        DEPS hyle_graph_route_cpu hyle_graph_route_apple hyle_graph_progress)
      hyle_stage_runtime(hyle_graph_route_apple_paged_consumer_test)
    endif()
  endif()
endif()

if(_HYLE_COMPUTE_HAVE_APPLE)

hyle_add_test(graph_route_apple NAME sampling_consumer
  SOURCES ../compute_route_apple/sampling_consumer.cc)
target_compile_definitions(hyle_graph_route_apple_sampling_consumer_test PRIVATE HYLE_G08_GRAPH=1)
hyle_stage_runtime(hyle_graph_route_apple_sampling_consumer_test)
if(_HYLE_COMPUTE_REQUEST_CPU AND TARGET hyle_graph_route_cpu AND TARGET hyle_graph_progress)
  hyle_add_test(graph_route_apple NAME native_effects
    SOURCES ../graph/route_effects_test.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress hyle_compute_backend_cpu
      hyle_compute_provider_native hyle_compute_provider_ggml)
  target_include_directories(hyle_graph_route_apple_native_effects_test PRIVATE
    "${HYLE_SOURCE_DIR}/src/graph" "${HYLE_SOURCE_DIR}/src/compute")
  set_tests_properties(hyle_graph_route_apple_native_effects_test PROPERTIES
    ENVIRONMENT "GTEST_FILTER=FoundationEffects.*")
  hyle_stage_runtime(hyle_graph_route_apple_native_effects_test)
  hyle_add_test(graph_route_apple NAME public_effects
    SOURCES ../graph/public_effect_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_apple_public_effects_test)
  hyle_add_test(graph_route_apple NAME mixed_packed_weights
    SOURCES ../graph_route_cpu_apple/packed_weight_test.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_apple_mixed_packed_weights_test)
  hyle_add_test(graph_route_apple NAME mixed_contracts
    SOURCES ../graph/cpu_apple_contracts.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_apple_mixed_contracts_test)
  hyle_add_test(graph_route_apple NAME mixed_public_consumer
    SOURCES ../../samples/package/graph/cpu_apple/main.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_apple_mixed_public_consumer_test)
  hyle_add_test(graph_route_apple NAME mixed_finite
    SOURCES ../graph/route_symmetry_apple.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_apple_mixed_finite_test)
  hyle_add_test(graph_route_apple NAME completion_publication
    SOURCES ../graph/cpu_completion_publication.cc
    DEFINES HYLE_COMPLETION_PUBLICATION_APPLE=1
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_apple_completion_publication_test)
  hyle_add_test(graph_route_apple NAME mixed_pipeline
    SOURCES ../graph/native_joint_pipeline.cc
    DEFINES HYLE_G09_APPLE=1
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_apple_mixed_pipeline_test)
  set_tests_properties(hyle_graph_route_apple_mixed_pipeline_test PROPERTIES TIMEOUT 600)
  hyle_add_test(graph_route_apple NAME mixed_sampling
    SOURCES ../graph_route_cpu_apple/sampling_consumer.cc
    DEPS hyle_graph_route_cpu hyle_graph_progress)
  hyle_stage_runtime(hyle_graph_route_apple_mixed_sampling_test)
endif()
endif()

if(_HYLE_COMPUTE_HAVE_APPLE AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_apple NAME ga_indexer
    SOURCES ga_indexer_state_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_apple_ga_indexer_test)
  hyle_add_test(graph_route_apple NAME ga_ratio4
    SOURCES ga_ratio4_producer_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_apple_ga_ratio4_test)
  hyle_add_test(graph_route_apple NAME ga_mhc
    SOURCES ga_mhc_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_apple_ga_mhc_test)
endif()
