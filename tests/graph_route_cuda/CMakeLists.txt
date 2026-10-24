hyle_add_header_self_contained_test(graph_route_cuda
  HEADER_DIR graph HEADERS cuda.h)
if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_test(graph_route_cuda NAME float8_quantize_output
    SOURCES ../compute_models/float8_quantize_output_test.cc
    DEFINES HYLE_TEST_FP8_QUANT_CUDA=1 DEPS hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_float8_quantize_output_test)
  hyle_add_test(graph_route_cuda SOURCES grouped_fp8_power2_test.cc e8m0_weight_import_test.cc mlx_orientation_test.cc mlx_head_payload_test.cc compact_mla_e4_test.cc paged_e4_kv_test.cc fp8_channel_token_qdq_test.cc nvfp4_reference_test.cc grouped_nvfp4_reference_test.cc local_nvfp4_reference_test.cc graph_test.cc artifact_test.cc d5_qwen35_profile_test.cc d5_qwen35_gguf_fragment_test.cc d5_qwen_gguf_vocabulary_test.cc tensor_test.cc norm_test.cc sglang_primitives_test.cc
    ct_block128_activation_edges_test.cc ct_block128_qdq_test.cc blockwise_w4a8_test.cc mxfp4_reference_test.cc expert_working_set_test.cc
    reduction_test.cc topk_test.cc dspark_feedback_consumer_test.cc concat_test.cc gather_test.cc scatter_test.cc scan_test.cc ggml_test.cc
    stable_silu_test.cc attention_test.cc collectives_test.cc ga_dense_indexer_test.cc fp8_cast_test.cc
    DEPS hyle_graph_batch hyle_compute_provider_flashinfer hyle_compute_backend_cuda)
  target_link_libraries(hyle_graph_route_cuda_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cuda_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime(hyle_graph_route_cuda_test)
  hyle_add_test(graph_route_cuda NAME packed_consumer
    SOURCES packed_resident_consumer.cc DEPS hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_packed_consumer_test)
  hyle_add_test(graph_route_cuda NAME gguf_nvfp4
    SOURCES gguf_nvfp4_test.cc DEPS hyle_compute_backend_cuda)
  target_link_libraries(hyle_graph_route_cuda_gguf_nvfp4_test PRIVATE ggml)
  hyle_stage_runtime(hyle_graph_route_cuda_gguf_nvfp4_test)
  hyle_add_test(graph_route_cuda NAME grouped_shared_activation
    SOURCES grouped_shared_activation_test.cc DEPS hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_grouped_shared_activation_test)
  hyle_add_test(graph_route_cuda NAME ct_packed_linear
    SOURCES ct_packed_grouped_test.cc ct_packed_linear_test.cc gptq_reference_test.cc static_fp8_reference_test.cc scalar_float_test.cc DEPS hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_ct_packed_linear_test)
  hyle_add_test(graph_route_cuda NAME fp8_shared_prologue
    SOURCES fp8_shared_prologue_test.cc DEPS hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_fp8_shared_prologue_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA)

hyle_add_test(graph_route_cuda NAME sampling_consumer
  SOURCES ../compute_route_cuda/sampling_consumer.cc DEPS hyle_compute_backend_cuda)
target_compile_definitions(hyle_graph_route_cuda_sampling_consumer_test PRIVATE HYLE_G08_GRAPH=1)
hyle_stage_runtime(hyle_graph_route_cuda_sampling_consumer_test)

hyle_add_test(graph_route_cuda NAME qwen_sampling_graph
  SOURCES ../graph_route_cpu/qwen_sampling_graph_test.cc
  DEFINES HYLE_TEST_QWEN_SAMPLING_CUDA=1
  DEPS hyle_compute_backend_cuda)
hyle_stage_runtime(hyle_graph_route_cuda_qwen_sampling_graph_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cuda NAME ga_indexer
    SOURCES ga_indexer_state_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cuda_ga_indexer_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cuda NAME ga_ratio4
    SOURCES ga_ratio4_producer_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cuda_ga_ratio4_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cuda NAME ga_mhc
    SOURCES ga_mhc_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cuda_ga_mhc_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_test(graph_route_cuda NAME ga_encoded_bank
    SOURCES ga_encoded_bank_test.cc DEPS hyle_graph_batch hyle_compute_backend_cuda hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cuda_ga_encoded_bank_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cuda NAME ga_full_ffn
    SOURCES ga_full_ffn_test.cc DEPS hyle_graph_batch hyle_compute_state hyle_simd hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_ga_full_ffn_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cuda NAME glm_flash_mixed_mlx
    SOURCES glm_flash_mixed_mlx_test.cc DEPS hyle_graph_batch hyle_compute_state hyle_simd hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_glm_flash_mixed_mlx_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_test(graph_route_cuda NAME fp8_outer_scale
    SOURCES fp8_outer_scale_test.cc DEPS hyle_simd hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_fp8_outer_scale_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_test(graph_route_cuda NAME fp8_scale_policy
    SOURCES fp8_scale_policy_test.cc DEPS hyle_simd hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_fp8_scale_policy_test)
endif()

if(_HYLE_COMPUTE_HAVE_CUDA)
  hyle_add_test(graph_route_cuda NAME dspark_original_confidence
    SOURCES dspark_original_confidence_test.cc DEPS hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_dspark_original_confidence_test)
  hyle_add_test(graph_route_cuda NAME dspark_complete
    SOURCES dspark_complete_consumer_test.cc DEPS hyle_simd hyle_compute_backend_cuda)
  target_link_libraries(hyle_graph_route_cuda_dspark_complete_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cuda_dspark_complete_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime(hyle_graph_route_cuda_dspark_complete_test)
  hyle_add_test(graph_route_cuda NAME glm_original_mtp_input
    SOURCES glm_original_mtp_input_test.cc DEPS hyle_compute_backend_cuda)
  hyle_stage_runtime(hyle_graph_route_cuda_glm_original_mtp_input_test)
  hyle_add_test(graph_route_cuda NAME qwen27_original_mtp_input
    SOURCES ../compute_models/qwen27_original_mtp_input_test.cc DEPS hyle_simd hyle_compute_backend_cuda)
  target_compile_definitions(hyle_graph_route_cuda_qwen27_original_mtp_input_test PRIVATE HYLE_TEST_Q27_CUDA=1)
  target_link_libraries(hyle_graph_route_cuda_qwen27_original_mtp_input_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cuda_qwen27_original_mtp_input_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime(hyle_graph_route_cuda_qwen27_original_mtp_input_test)
  hyle_add_test(graph_route_cuda NAME dspark_target_binding
    SOURCES ../compute_models/dspark_target_binding_test.cc DEPS hyle_simd hyle_compute_backend_cuda)
  target_compile_definitions(hyle_graph_route_cuda_dspark_target_binding_test PRIVATE HYLE_TEST_DSPARK_TARGET_CUDA=1)
  target_link_libraries(hyle_graph_route_cuda_dspark_target_binding_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cuda_dspark_target_binding_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime(hyle_graph_route_cuda_dspark_target_binding_test)
  if(TARGET hyle_compute_state)
    hyle_add_test(graph_route_cuda NAME qwen24_original_mtp_kv
      SOURCES ../compute_models/qwen24_original_mtp_kv_test.cc DEPS hyle_compute_state hyle_simd hyle_compute_backend_cuda)
    target_compile_definitions(hyle_graph_route_cuda_qwen24_original_mtp_kv_test PRIVATE HYLE_TEST_Q24_CUDA=1)
    target_link_libraries(hyle_graph_route_cuda_qwen24_original_mtp_kv_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cuda_qwen24_original_mtp_kv_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cuda_qwen24_original_mtp_kv_test)
    hyle_add_test(graph_route_cuda NAME qwen_mtp_state_consumer
      SOURCES ../compute_models/qwen_mtp_state_consumer_test.cc DEPS hyle_compute_state hyle_simd hyle_compute_backend_cuda)
    target_compile_definitions(hyle_graph_route_cuda_qwen_mtp_state_consumer_test PRIVATE HYLE_TEST_QWEN_STATE_CUDA=1)
    target_link_libraries(hyle_graph_route_cuda_qwen_mtp_state_consumer_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cuda_qwen_mtp_state_consumer_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cuda_qwen_mtp_state_consumer_test)
    hyle_add_test(graph_route_cuda NAME glm_mtp_index_carry
      SOURCES ../compute_models/glm_mtp_index_carry_test.cc DEPS hyle_compute_state hyle_simd hyle_compute_backend_cuda)
    target_compile_definitions(hyle_graph_route_cuda_glm_mtp_index_carry_test PRIVATE HYLE_TEST_GLM_CARRY_CUDA=1)
    target_link_libraries(hyle_graph_route_cuda_glm_mtp_index_carry_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cuda_glm_mtp_index_carry_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cuda_glm_mtp_index_carry_test)
    hyle_add_test(graph_route_cuda NAME glm_original_mtp_chain
      SOURCES ../compute_models/glm_original_mtp_chain_test.cc DEPS hyle_simd hyle_compute_backend_cuda)
    target_compile_definitions(hyle_graph_route_cuda_glm_original_mtp_chain_test PRIVATE HYLE_TEST_GLM_ORIGINAL_CHAIN_CUDA=1)
    target_link_libraries(hyle_graph_route_cuda_glm_original_mtp_chain_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cuda_glm_original_mtp_chain_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cuda_glm_original_mtp_chain_test)
    hyle_add_test(graph_route_cuda NAME glm_mtp_state_consumer
      SOURCES ../compute_models/glm_mtp_state_consumer_test.cc
              ../compute_models/glm_pro_mtp_state_consumer_test.cc
              ../compute_models/glm_pro_original_early_history_test.cc
      DEPS hyle_compute_state hyle_simd hyle_compute_backend_cuda)
    target_compile_definitions(hyle_graph_route_cuda_glm_mtp_state_consumer_test PRIVATE HYLE_TEST_GLM_STATE_CUDA=1)
    target_link_libraries(hyle_graph_route_cuda_glm_mtp_state_consumer_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cuda_glm_mtp_state_consumer_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cuda_glm_mtp_state_consumer_test)
    hyle_add_test(graph_route_cuda NAME glm_official_index_cache
      SOURCES ../compute_models/glm_official_index_cache_test.cc DEPS hyle_compute_state hyle_simd hyle_compute_backend_cuda)
    target_compile_definitions(hyle_graph_route_cuda_glm_official_index_cache_test PRIVATE HYLE_TEST_GLM_CACHE_CUDA=1)
    target_link_libraries(hyle_graph_route_cuda_glm_official_index_cache_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cuda_glm_official_index_cache_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cuda_glm_official_index_cache_test)
    hyle_add_test(graph_route_cuda NAME dspark_main_history
      SOURCES ../compute_models/dspark_main_history_test.cc DEPS hyle_compute_state hyle_simd hyle_compute_backend_cuda)
    target_compile_definitions(hyle_graph_route_cuda_dspark_main_history_test PRIVATE HYLE_TEST_DSPARK_HISTORY_CUDA=1)
    target_link_libraries(hyle_graph_route_cuda_dspark_main_history_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cuda_dspark_main_history_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cuda_dspark_main_history_test)
    hyle_add_test(graph_route_cuda NAME dspark_original_state_consumer
      SOURCES ../compute_models/dspark_original_state_consumer_test.cc
              ../compute_models/dspark_original_early_history_test.cc
              ../compute_models/dspark_pro_mtp_state_consumer_test.cc
      DEPS hyle_compute_state hyle_simd hyle_compute_backend_cuda)
    target_compile_definitions(hyle_graph_route_cuda_dspark_original_state_consumer_test PRIVATE HYLE_TEST_DSPARK_STATE_CUDA=1)
    target_link_libraries(hyle_graph_route_cuda_dspark_original_state_consumer_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cuda_dspark_original_state_consumer_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cuda_dspark_original_state_consumer_test)
  endif()
  if(TARGET hyle_graph_route_cpu AND TARGET hyle_compute_state AND HYLE_GGML_CPU_ENABLED)
    get_target_property(_hyle_dspark_cpu_capabilities hyle_compute_provider_native HYLE_COMPUTE_CAPABILITIES)
    if("torch_cpu" IN_LIST _hyle_dspark_cpu_capabilities)
      hyle_add_test(graph_route_cuda NAME dspark_journey
        SOURCES dspark_journey_consumer_test.cc
        DEPS hyle_graph_route_cpu hyle_compute_state hyle_simd hyle_compute_backend_cuda)
      target_link_libraries(hyle_graph_route_cuda_dspark_journey_test PRIVATE ggml)
      target_include_directories(hyle_graph_route_cuda_dspark_journey_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
      hyle_stage_runtime(hyle_graph_route_cuda_dspark_journey_test)
    endif()
    unset(_hyle_dspark_cpu_capabilities)
  endif()
endif()
