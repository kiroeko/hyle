hyle_add_header_self_contained_test(graph_route_cpu
  HEADER_DIR graph HEADERS cpu.h)
hyle_add_test(graph_route_cpu SOURCES grouped_fp8_power2_test.cc e8m0_weight_import_test.cc mlx_orientation_test.cc mlx_head_payload_test.cc compact_mla_e4_test.cc ct_block128_activation_edges_test.cc ct_block128_qdq_test.cc ct_packed_grouped_test.cc blockwise_w4a8_test.cc mxfp4_reference_test.cc local_nvfp4_reference_test.cc static_fp8_reference_test.cc fp8_channel_token_qdq_test.cc nvfp4_reference_test.cc grouped_nvfp4_reference_test.cc gptq_reference_test.cc ct_packed_linear_test.cc
  ../compute_route_cpu/current_rules_test.cc
  ../compute_route_cpu/cpu_expanded_gguf_test.cc
  ../compute_route_cpu/cpu_mlx_quantized_test.cc
  ../compute_route_cpu/cpu_artifact_decode_test.cc
  ../compute_route_cpu/ds4_roundtrip_test.cc
  ../compute_route_cpu/context_test.cc d5_qwen35_profile_test.cc
  d5_qwen_gguf_vocabulary_test.cc
  d5_qwen38_official_test.cc hybrid_ffn_numeric_test.cc
  d5_qwen35_gguf_projection_test.cc d5_qwen35_gguf_fragment_test.cc graph_test.cc concat_test.cc
  DEPS hyle_compute_provider_native hyle_compute_provider_onednn
       hyle_graph_provider_onednn hyle_compute_provider_ggml)
target_compile_definitions(hyle_graph_route_cpu_test PRIVATE
  HYLE_TEST_GRAPH_ROUTE_CPU=1)
get_target_property(_hyle_native_capabilities hyle_compute_provider_native
  HYLE_COMPUTE_CAPABILITIES)
if("torch_cpu" IN_LIST _hyle_native_capabilities)
  target_compile_definitions(hyle_graph_route_cpu_test PRIVATE HYLE_TEST_TORCH_CPU=1)
  hyle_add_test(graph_route_cpu NAME float8_quantize_output
    SOURCES ../compute_models/float8_quantize_output_test.cc
    DEFINES HYLE_TEST_FP8_QUANT_CPU=1 DEPS hyle_compute_provider_native)
  hyle_stage_runtime(hyle_graph_route_cpu_float8_quantize_output_test)
endif()
if(HYLE_GGML_CPU_ENABLED)
  target_sources(hyle_graph_route_cpu_test PRIVATE paged_e4_kv_test.cc)
  target_link_libraries(hyle_graph_route_cpu_test PRIVATE hyle_graph_batch)
  target_link_libraries(hyle_graph_route_cpu_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cpu_test SYSTEM PRIVATE
    "${HYLE_GGML_SOURCE_DIR}/src")
endif()
hyle_stage_runtime(hyle_graph_route_cpu_test)

if(HYLE_GGML_CPU_ENABLED)
  hyle_add_test(graph_route_cpu NAME qwen27_original_mtp_input
    SOURCES ../compute_models/qwen27_original_mtp_input_test.cc DEPS hyle_simd)
  target_compile_definitions(hyle_graph_route_cpu_qwen27_original_mtp_input_test PRIVATE HYLE_TEST_Q27_CPU=1)
  target_link_libraries(hyle_graph_route_cpu_qwen27_original_mtp_input_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cpu_qwen27_original_mtp_input_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime(hyle_graph_route_cpu_qwen27_original_mtp_input_test)
  hyle_add_test(graph_route_cpu NAME dspark_target_binding
    SOURCES ../compute_models/dspark_target_binding_test.cc DEPS hyle_simd)
  target_compile_definitions(hyle_graph_route_cpu_dspark_target_binding_test PRIVATE HYLE_TEST_DSPARK_TARGET_CPU=1)
  target_link_libraries(hyle_graph_route_cpu_dspark_target_binding_test PRIVATE ggml)
  target_include_directories(hyle_graph_route_cpu_dspark_target_binding_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
  hyle_stage_runtime(hyle_graph_route_cpu_dspark_target_binding_test)
  if(TARGET hyle_compute_state)
    hyle_add_test(graph_route_cpu NAME qwen24_original_mtp_kv
      SOURCES ../compute_models/qwen24_original_mtp_kv_test.cc DEPS hyle_compute_state hyle_simd)
    target_compile_definitions(hyle_graph_route_cpu_qwen24_original_mtp_kv_test PRIVATE HYLE_TEST_Q24_CPU=1)
    target_link_libraries(hyle_graph_route_cpu_qwen24_original_mtp_kv_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cpu_qwen24_original_mtp_kv_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cpu_qwen24_original_mtp_kv_test)
    hyle_add_test(graph_route_cpu NAME qwen_mtp_state_consumer
      SOURCES ../compute_models/qwen_mtp_state_consumer_test.cc DEPS hyle_compute_state hyle_simd)
    target_compile_definitions(hyle_graph_route_cpu_qwen_mtp_state_consumer_test PRIVATE HYLE_TEST_QWEN_STATE_CPU=1)
    target_link_libraries(hyle_graph_route_cpu_qwen_mtp_state_consumer_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cpu_qwen_mtp_state_consumer_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cpu_qwen_mtp_state_consumer_test)
    hyle_add_test(graph_route_cpu NAME glm_mtp_index_carry
      SOURCES ../compute_models/glm_mtp_index_carry_test.cc DEPS hyle_compute_state hyle_simd)
    target_compile_definitions(hyle_graph_route_cpu_glm_mtp_index_carry_test PRIVATE HYLE_TEST_GLM_CARRY_CPU=1)
    target_link_libraries(hyle_graph_route_cpu_glm_mtp_index_carry_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cpu_glm_mtp_index_carry_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cpu_glm_mtp_index_carry_test)
    hyle_add_test(graph_route_cpu NAME glm_original_mtp_chain
      SOURCES ../compute_models/glm_original_mtp_chain_test.cc DEPS hyle_simd)
    target_compile_definitions(hyle_graph_route_cpu_glm_original_mtp_chain_test PRIVATE HYLE_TEST_GLM_ORIGINAL_CHAIN_CPU=1)
    target_link_libraries(hyle_graph_route_cpu_glm_original_mtp_chain_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cpu_glm_original_mtp_chain_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cpu_glm_original_mtp_chain_test)
    hyle_add_test(graph_route_cpu NAME glm_mtp_state_consumer
      SOURCES ../compute_models/glm_mtp_state_consumer_test.cc
              ../compute_models/glm_pro_mtp_state_consumer_test.cc
      DEPS hyle_compute_state hyle_simd)
    target_compile_definitions(hyle_graph_route_cpu_glm_mtp_state_consumer_test PRIVATE HYLE_TEST_GLM_STATE_CPU=1)
    target_link_libraries(hyle_graph_route_cpu_glm_mtp_state_consumer_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cpu_glm_mtp_state_consumer_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cpu_glm_mtp_state_consumer_test)
    hyle_add_test(graph_route_cpu NAME glm_official_index_cache
      SOURCES ../compute_models/glm_official_index_cache_test.cc DEPS hyle_compute_state hyle_simd)
    target_compile_definitions(hyle_graph_route_cpu_glm_official_index_cache_test PRIVATE HYLE_TEST_GLM_CACHE_CPU=1)
    target_link_libraries(hyle_graph_route_cpu_glm_official_index_cache_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cpu_glm_official_index_cache_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cpu_glm_official_index_cache_test)
    hyle_add_test(graph_route_cpu NAME dspark_main_history
      SOURCES ../compute_models/dspark_main_history_test.cc DEPS hyle_compute_state hyle_simd)
    target_compile_definitions(hyle_graph_route_cpu_dspark_main_history_test PRIVATE HYLE_TEST_DSPARK_HISTORY_CPU=1)
    target_link_libraries(hyle_graph_route_cpu_dspark_main_history_test PRIVATE ggml)
    target_include_directories(hyle_graph_route_cpu_dspark_main_history_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
    hyle_stage_runtime(hyle_graph_route_cpu_dspark_main_history_test)
    if("torch_cpu" IN_LIST _hyle_native_capabilities)
      hyle_add_test(graph_route_cpu NAME dspark_original_state_consumer
        SOURCES ../compute_models/dspark_original_state_consumer_test.cc
                ../compute_models/dspark_pro_mtp_state_consumer_test.cc
        DEPS hyle_compute_state hyle_simd)
      target_compile_definitions(hyle_graph_route_cpu_dspark_original_state_consumer_test PRIVATE HYLE_TEST_DSPARK_STATE_CPU=1)
      target_link_libraries(hyle_graph_route_cpu_dspark_original_state_consumer_test PRIVATE ggml)
      target_include_directories(hyle_graph_route_cpu_dspark_original_state_consumer_test SYSTEM PRIVATE "${HYLE_GGML_SOURCE_DIR}/src")
      hyle_stage_runtime(hyle_graph_route_cpu_dspark_original_state_consumer_test)
    endif()
  endif()
endif()

# The installed/source consumer is also run in the targeted route test suite.
# Sched is a test dependency, not a shipping dependency of the CPU route.
hyle_add_test(graph_route_cpu NAME cpu_executor
  SOURCES ../../samples/package/graph/cpu_executor/main.cc
  DEPS hyle_sched hyle_compute_provider_native hyle_graph_provider_onednn)
hyle_stage_runtime(hyle_graph_route_cpu_cpu_executor_test)

hyle_add_test(graph_route_cpu NAME sampling_consumer
  SOURCES ../compute_route_cpu/sampling_consumer.cc DEPS hyle_compute_provider_native hyle_compute_provider_ggml hyle_compute_provider_onednn hyle_graph_provider_onednn)
target_compile_definitions(hyle_graph_route_cpu_sampling_consumer_test PRIVATE HYLE_G08_GRAPH=1)
hyle_stage_runtime(hyle_graph_route_cpu_sampling_consumer_test)

hyle_add_test(graph_route_cpu NAME gguf_nvfp4
  SOURCES ../compute_route_cpu/gguf_nvfp4_test.cc
  DEPS hyle_compute_provider_native hyle_compute_provider_onednn
  hyle_graph_provider_onednn hyle_compute_provider_ggml)
target_compile_definitions(hyle_graph_route_cpu_gguf_nvfp4_test PRIVATE
  HYLE_TEST_GRAPH_ROUTE_CPU=1)
hyle_stage_runtime(hyle_graph_route_cpu_gguf_nvfp4_test)
hyle_add_test(graph_route_cpu NAME grouped_shared_activation
  SOURCES ../compute_route_cpu/grouped_shared_activation_test.cc
  DEPS hyle_compute_provider_native hyle_compute_provider_onednn
  hyle_graph_provider_onednn hyle_compute_provider_ggml)
target_compile_definitions(hyle_graph_route_cpu_grouped_shared_activation_test PRIVATE
  HYLE_TEST_GRAPH_ROUTE_CPU=1)
hyle_stage_runtime(hyle_graph_route_cpu_grouped_shared_activation_test)

if(HYLE_GGML_CPU_ENABLED)
  target_compile_definitions(hyle_graph_route_cpu_test PRIVATE HYLE_TEST_GGML_CPU=1)
endif()

if(_HYLE_COMPUTE_HAVE_CPU AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cpu NAME ga_indexer
    SOURCES ga_indexer_state_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cpu_ga_indexer_test)
endif()

if(_HYLE_COMPUTE_HAVE_CPU AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cpu NAME ga_ratio4
    SOURCES ga_ratio4_producer_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cpu_ga_ratio4_test)
endif()

if(_HYLE_COMPUTE_HAVE_CPU AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cpu NAME ga_mhc
    SOURCES ga_mhc_test.cc DEPS hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cpu_ga_mhc_test)
endif()

if(_HYLE_COMPUTE_HAVE_CPU AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cpu NAME ga_full_ffn
    SOURCES ga_full_ffn_test.cc DEPS hyle_graph_batch hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cpu_ga_full_ffn_test)
endif()

if(_HYLE_COMPUTE_HAVE_CPU AND TARGET hyle_compute_state)
  hyle_add_test(graph_route_cpu NAME glm_flash_mixed_mlx
    SOURCES glm_flash_mixed_mlx_test.cc DEPS hyle_graph_batch hyle_compute_state hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cpu_glm_flash_mixed_mlx_test)
endif()

if(_HYLE_COMPUTE_HAVE_CPU)
  hyle_add_test(graph_route_cpu NAME fp8_outer_scale
    SOURCES fp8_outer_scale_test.cc DEPS hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cpu_fp8_outer_scale_test)
endif()

if(_HYLE_COMPUTE_HAVE_CPU)
  hyle_add_test(graph_route_cpu NAME fp8_scale_policy
    SOURCES fp8_scale_policy_test.cc DEPS hyle_simd)
  hyle_stage_runtime(hyle_graph_route_cpu_fp8_scale_policy_test)
endif()
