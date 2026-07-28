# compute-01: device/stream/event + completion. Always buildable (host fallback);
# the GPU-specific test self-skips at runtime when no CUDA device is present. The
# test TU is pure C++ (all CUDA is inside hyle_compute), so it needs no nvcc.
# compute-02: owned backend buffer + offset copy + explicit host mapping/coherence.
# Headline tests = KV-cache repeated offset writes and direct HostBuffer mapping.
hyle_add_test(compute SOURCES device_test.cc buffer_test.cc device_pool_test.cc kernel_test.cc contract_test.cc host_ops_test.cc)
# compute-06: self-authored SPIR-V compute kernel through the Vulkan backend (saxpy
# end-to-end + batched). GPU-skips at runtime if no Vulkan loader/device. The co_await
# completion case (#21) is #if HYLE_HAS_FCONTEXT-gated inside the TU, so link hyle_sched
# when fcontext exists (else the base sync tests still build with just hyle_mem).
if(HYLE_COMPUTE_VULKAN_ENABLED)
  # Compile the test shaders from source (tools/shaders/*.comp) to embedded SpirvBlob headers
  # via tools/hyle_shader.py -- reproducible from source, and the SAME tool a user runs. The
  # symbols live in namespace hyle_test_shaders (the tool's --namespace) so they can't clash.
  # saxpy.spv.h feeds the end-to-end/batch/timing tests; saxpy_spec.spv.h drives the #2/#3
  # spec-constant test (its local_size + SCALE are specialization constants).
  hyle_embed_shader(saxpy.comp saxpy.spv.h kSaxpySpirv hyle_test_shaders)
  hyle_embed_shader(saxpy_spec.comp saxpy_spec.spv.h kSaxpySpecSpirv hyle_test_shaders)
  if(HYLE_HAS_FCONTEXT)
    hyle_add_test(compute NAME vulkan SOURCES vulkan_test.cc
                  "${HYLE_SHADER_kSaxpySpirv}" "${HYLE_SHADER_kSaxpySpecSpirv}"
                  DEFINES HYLE_HAS_FCONTEXT DEPS hyle_mem hyle_sched hyle_platform)
  else()
    hyle_add_test(compute NAME vulkan SOURCES vulkan_test.cc
                  "${HYLE_SHADER_kSaxpySpirv}" "${HYLE_SHADER_kSaxpySpecSpirv}"
                  DEPS hyle_mem hyle_platform)
  endif()
  target_include_directories(hyle_compute_vulkan_test PRIVATE "${HYLE_SHADER_DIR}")
  if(HYLE_TEST_INSTRUMENTATION)
    # Second ctest of the SAME exe with HYLE_VK_FORCE_STAGING=1 (a TEST-ONLY env hook,
    # only honored under HYLE_TESTING): forces the non-ReBAR STAGING path so it's covered
    # on ReBAR dev machines (RTX 4070) where the default direct-map path would otherwise
    # hide it. Same VulkanCompute cases exercise vk_buffer_set/get/copy through the staging
    # double-buffer + copyBuffer barrier chain instead of the direct map.
    add_test(NAME hyle_compute_vulkan_staging_test
             COMMAND hyle_compute_vulkan_test --gtest_filter=VulkanCompute.*)
    set_tests_properties(hyle_compute_vulkan_staging_test PROPERTIES
                         ENVIRONMENT "HYLE_VK_FORCE_STAGING=1"
                         LABELS "compute")
    # Best-effort non-coherent HOST_CACHED variant. The test queries the selected memory
    # type and skips honestly when the device exposes only coherent host-visible memory.
    add_test(NAME hyle_compute_vulkan_noncoherent_test
             COMMAND hyle_compute_vulkan_test
                     --gtest_filter=VulkanCompute.NonCoherentHostBufferRoundTrip)
    set_tests_properties(hyle_compute_vulkan_noncoherent_test PROPERTIES
                         ENVIRONMENT "HYLE_VK_FORCE_NONCOHERENT=1"
                         LABELS "compute")
    # #1 pipeline cache 落盘: use one build-dir file and run twice via a fixture. The
    # first run proves save; the second loads and validates vendor/device/UUID before saxpy.
    add_test(NAME hyle_compute_vulkan_cache_seed_test
             COMMAND hyle_compute_vulkan_test --gtest_filter=VulkanCompute.*)
    set_tests_properties(hyle_compute_vulkan_cache_seed_test PROPERTIES
                         ENVIRONMENT "HYLE_VK_PIPELINE_CACHE=${CMAKE_CURRENT_BINARY_DIR}/hyle_vk_缓存.bin"
                         FIXTURES_SETUP VK_PIPE_CACHE
                         LABELS "compute")
    add_test(NAME hyle_compute_vulkan_cache_load_test
             COMMAND hyle_compute_vulkan_test --gtest_filter=VulkanCompute.*)
    set_tests_properties(hyle_compute_vulkan_cache_load_test PROPERTIES
                         ENVIRONMENT "HYLE_VK_PIPELINE_CACHE=${CMAKE_CURRENT_BINARY_DIR}/hyle_vk_缓存.bin"
                         FIXTURES_REQUIRED VK_PIPE_CACHE
                         LABELS "compute")
  endif()
endif()
if(TARGET hyle_sched)
  # compute-03: co_await GPU stream completion -> fiber wake. Needs sched (the poller
  # wakes a fiber via Waker) -- only where fcontext exists, like the io co_await tests.
  hyle_add_test(compute NAME gpu_await SOURCES gpu_await_test.cc DEPS hyle_sched hyle_mem)
  # compute-05: ggml_bridge -- borrow ggml operators on HYLE VRAM + spawn_blocking
  # completion + fiber scheduling. Needs sched (spawn_blocking/JoinHandle) AND ggml
  # fetched (HYLE_COMPUTE_GGML). Same gate as src/compute's ggml_bridge.cc.
  if(HYLE_COMPUTE_GGML)
    hyle_add_test(compute NAME ggml_bridge SOURCES ggml_bridge_test.cc DEPS hyle_sched hyle_mem)
  endif()
endif()
