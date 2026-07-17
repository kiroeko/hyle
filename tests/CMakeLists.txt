include(FetchContent)

# GoogleTest (charter: unit test framework). Pinned tag, FetchContent GIT form (same
# open-box-auto-download contract as ggml): first `cmake` fetches it into .deps/
# googletest-src ONCE (FETCHCONTENT_BASE_DIR=.deps, set in root), reused across build
# wipes. No pre-placed mirror.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2
  GIT_SHALLOW    TRUE)
# Use shared CRT on MSVC to match our build.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

hyle_add_test(core SOURCES core/scaffolding_test.cc)
hyle_add_test(error SOURCES error/status_test.cc error/error_handler_test.cc)
hyle_add_test(obs SOURCES obs/log_test.cc)
hyle_add_test(platform SOURCES platform/wait_test.cc platform/thread_test.cc platform/dynlib_test.cc platform/fs_test.cc)
hyle_add_test(mem SOURCES mem/allocator_test.cc mem/memory_arena_test.cc mem/pool_stack_test.cc mem/alignment_test.cc mem/pmr_test.cc mem/growable_pool_test.cc DEPS hyle_platform)
hyle_add_test(util SOURCES util/ref_counted_test.cc util/scope_guard_test.cc util/type_name_test.cc util/registry_test.cc util/random_test.cc util/spin_lock_test.cc util/demangle_test.cc DEPS hyle_container)
hyle_add_test(time SOURCES time/clock_test.cc time/timer_wheel_test.cc time/common_timeout_queue_test.cc DEPS hyle_platform)
hyle_add_test(io SOURCES io/proactor_test.cc)
if(WIN32)
  # The io test itself calls Winsock directly (WSASocketW/bind/getsockname/... in
  # the IOCP TCP round-trip test), so it needs ws2_32 linked into the test exe.
  target_link_libraries(hyle_io_test PRIVATE ws2_32)
endif()
hyle_add_test(math SOURCES math/scalar_test.cc)
# compute-01: device/stream/event + completion. Always buildable (host fallback);
# the GPU-specific test self-skips at runtime when no CUDA device is present. The
# test TU is pure C++ (all CUDA is inside hyle_compute), so it needs no nvcc.
# compute-02: device buffer (alloc/set/get(offset)/copy/clear + pinned staging),
# headline test = KV-cache pattern (large buffer repeated offset writes).
# compute-02b: pinned staging via mem/ (pinned_allocator + mem::MemoryArena/Arena)
# + PinnedStaging batched submit -- needs hyle_mem for the pool/arena in the test.
hyle_add_test(compute SOURCES compute/device_test.cc compute/buffer_test.cc compute/pinned_pool_test.cc compute/device_pool_test.cc compute/kernel_test.cc compute/contract_test.cc compute/host_ops_test.cc DEPS hyle_mem)
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
    hyle_add_test(compute NAME vulkan SOURCES compute/vulkan_test.cc
                  "${HYLE_SHADER_kSaxpySpirv}" "${HYLE_SHADER_kSaxpySpecSpirv}"
                  DEFINES HYLE_HAS_FCONTEXT DEPS hyle_mem hyle_sched)
  else()
    hyle_add_test(compute NAME vulkan SOURCES compute/vulkan_test.cc
                  "${HYLE_SHADER_kSaxpySpirv}" "${HYLE_SHADER_kSaxpySpecSpirv}" DEPS hyle_mem)
  endif()
  target_include_directories(hyle_compute_vulkan_test PRIVATE "${HYLE_SHADER_DIR}")
  # Second ctest of the SAME exe with HYLE_VK_FORCE_STAGING=1 (a TEST-ONLY env hook, only
  # honored under HYLE_TESTING): forces the non-ReBAR STAGING path so it's covered on ReBAR
  # dev machines (RTX 4070) where the default direct-map path would otherwise hide it. Same
  # VulkanCompute cases, exercising vk_buffer_set/get/copy through the staging double-buffer
  # + copyBuffer barrier chain instead of the direct map.
  add_test(NAME hyle_compute_vulkan_staging_test
           COMMAND hyle_compute_vulkan_test --gtest_filter=VulkanCompute.*)
  set_tests_properties(hyle_compute_vulkan_staging_test PROPERTIES
                       ENVIRONMENT "HYLE_VK_FORCE_STAGING=1")
  # #1 pipeline cache 落盘: a variant with HYLE_VK_PIPELINE_CACHE pointing at a build-dir file
  # (opt-in persistence, TEST-ONLY under HYLE_TESTING). Run TWICE in sequence via a fixture:
  # the first run seeds the file from an empty cache (proving the SAVE path -- PipelineCachePersist
  # asserts a non-empty blob); the second run loads that file at VkContext init (the LOAD+UUID
  # path) and must still pass saxpy (a rejected/foreign cache would just be ignored, not fail).
  add_test(NAME hyle_compute_vulkan_cache_seed_test
           COMMAND hyle_compute_vulkan_test --gtest_filter=VulkanCompute.*)
  set_tests_properties(hyle_compute_vulkan_cache_seed_test PROPERTIES
                       ENVIRONMENT "HYLE_VK_PIPELINE_CACHE=${CMAKE_CURRENT_BINARY_DIR}/hyle_vk_pipe_cache.bin"
                       FIXTURES_SETUP VK_PIPE_CACHE)
  add_test(NAME hyle_compute_vulkan_cache_load_test
           COMMAND hyle_compute_vulkan_test --gtest_filter=VulkanCompute.*)
  set_tests_properties(hyle_compute_vulkan_cache_load_test PROPERTIES
                       ENVIRONMENT "HYLE_VK_PIPELINE_CACHE=${CMAKE_CURRENT_BINARY_DIR}/hyle_vk_pipe_cache.bin"
                       FIXTURES_REQUIRED VK_PIPE_CACHE)
endif()
hyle_add_test(container SOURCES container/bounded_queue_test.cc container/sharded_queue_test.cc container/work_stealing_deque_test.cc container/ws_fifo_queue_test.cc container/hash_test.cc container/carray_test.cc container/small_vector_test.cc container/probe_test.cc container/swiss_table_test.cc container/string_ref_test.cc container/string_buf_test.cc container/concurrent_vector_test.cc container/string_algo_test.cc container/path_test.cc DEPS hyle_platform mimalloc_static)
# graph-01 (step-1): task-dependency DAG incremental recompute. Pure CPU, serial evaluate
# -- no sched dep yet (parallel node execution is step-2).
hyle_add_test(graph SOURCES graph/taskgraph_test.cc DEPS hyle_graph)
# hash_test.cc includes the opt-in hyle/container/hash_inline.h, which pulls the
# vendored xxhash + wyhash single-headers into that TU -- add their include dirs.
target_include_directories(hyle_container_test SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/third_party/xxhash"
  "${PROJECT_SOURCE_DIR}/third_party/wyhash")
# simd: the scalar test (dispatch + vec floor) always builds. The vec op suite is ALSO
# compiled per real ISA backend (sse4 / avx2) -- SAME source, different HYLE_SIMD_TARGET_*
# + -march, each pairing the backend against the naive reference. GCC/Clang need the
# arch flag to enable the intrinsics; MSVC x64 has SSE4.2 unconditionally and uses
# /arch:AVX2 for the 256-bit build. These variant exes run on the build host, which must
# support the ISA (both dev platforms are AVX2 -- matches the variant-DLL test model).
hyle_add_test(simd SOURCES simd/dispatch_test.cc simd/vec_test.cc simd/native_test.cc
              simd/runtime_test.cc simd/ops_test.cc simd/search_test.cc simd/scan_test.cc simd/dtype_test.cc
              simd/math_test.cc simd/sort_test.cc simd/radix_sort_test.cc simd/bitpack_test.cc)
# vec_test compiled per ISA variant: same source, different HYLE_SIMD_TARGET_* + the
# matching codegen flags from the single-source hyle_isa_flags (no MSVC/else drift here).
# Arch-gated (mirrors src/simd's CMAKE_SYSTEM_PROCESSOR split): x86 builds the sse4/avx2
# variants, aarch64 builds the neon variant -- feeding -mavx2 to an ARM build (or the ARM
# neon.h to an x86 build) would fail. Each variant runs on the build host, so it must be
# the host's real ISA (the variant-DLL test model).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_isa_flags(_sse4_flags SSE42)
  hyle_isa_flags(_avx2_flags AVX2)
  # Fixed-width partial-vector proof: every logical width below the target native
  # width gets exact-byte LoadU/StoreU guards; every half-able width gets the scalar
  # half/combine oracle. Windows uses VirtualProtect guard pages, Unix uses mprotect.
  set(_fixed_partial_tests simd/vec_partial_boundary_test.cc simd/vec_half_combine_oracle_test.cc)
  hyle_add_test(simd NAME sse4 SOURCES simd/vec_test.cc simd/native_test.cc ${_fixed_partial_tests}
                DEFINES HYLE_SIMD_TARGET_SSE4 OPTIONS ${_sse4_flags})
  hyle_add_test(simd NAME avx2 SOURCES simd/vec_test.cc simd/native_test.cc ${_fixed_partial_tests}
                DEFINES HYLE_SIMD_TARGET_AVX2 OPTIONS ${_avx2_flags})
  # simd-08 档C crypto: the base sse4/avx2 variants above compile WITHOUT -maes/-mpclmul, so on
  # GCC/Clang the AES/CLMul op face takes the SOFTWARE fallback. This extra variant adds those
  # flags (SSE42_AES tier) so the AES-NI / PCLMULQDQ HARDWARE path is actually compiled + run
  # and its result checked against the same in-test FIPS-197 oracle -- proving hardware ==
  # software. (On MSVC both variants are identical since AES-NI is unconditional there, but the
  # variant is harmless and keeps one build matrix across compilers.)
  hyle_isa_flags(_sse4_aes_flags SSE42_AES)
  hyle_add_test(simd NAME sse4_aes SOURCES simd/vec_test.cc simd/native_test.cc DEFINES HYLE_SIMD_TARGET_SSE4 OPTIONS ${_sse4_aes_flags})
  # AVX-512 vec_test variant: exercises the avx512.h backend's FULL op face (incl. simd-08
  # crypto + simd-09 cmp128/intdiv/RNG that route through the 512-bit backend). Historically
  # avx512.h was "written + objdump-verified zmm but NEVER RUN" (no AVX-512 dev machine). This
  # variant + Intel SDE closes that: build it always (GCC/Clang emit AVX-512 codegen without
  # needing the host to support it), then RUN under `sde64 -future` on an AVX2-only box. CTest
  # can't auto-run it (would SIGILL on a non-AVX512 host), so it is NOT registered as an
  # auto-run ctest -- it is built for compile validation + manual SDE execution. The optional
  # HYLE_SIMD_AVX512_RUNNABLE gate marks the host as真 AVX-512 (or SDE-wrapped) to auto-run.
  hyle_isa_flags(_avx512_flags AVX512)
  hyle_add_test(simd NAME avx512 SOURCES simd/vec_test.cc simd/native_test.cc ${_fixed_partial_tests}
                DEFINES HYLE_SIMD_TARGET_AVX512 OPTIONS ${_avx512_flags})
  # AVX512_DL variant: same avx512.h backend, compiled with GFNI/VBMI/BITALG so the NATIVE
  # BitShuffle (VPSHUFBITQMB) / MultiRotateRight (VPMULTISHIFTQB) paths are taken instead of the
  # store+reload fallback. Built for compile validation + SDE execution (Ice Lake ISA); NOT
  # auto-run (SIGILLs on a Skylake-X / AVX2 host). Running BOTH avx512 and avx512_dl under SDE
  # verifies the fallback AND the native path agree with the scalar oracle. GCC/Clang only: MSVC's
  # /arch:AVX512 does not expose VBMI/BITALG macros (probe-verified), so the native path never
  # compiles on MSVC -- the base avx512 variant already covers the MSVC fallback, making a dl
  # variant there a pointless duplicate.
  if(NOT MSVC)
    hyle_isa_flags(_avx512dl_flags AVX512_DL)
    hyle_add_test(simd NAME avx512_dl SOURCES simd/vec_test.cc simd/native_test.cc DEFINES HYLE_SIMD_TARGET_AVX512 OPTIONS ${_avx512dl_flags})
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_isa_flags(_neon_flags NEON)
  # The same fixed-width partial proof covers NEON's physical D/Q split. All cases
  # are permanent default-on regressions.
  hyle_add_test(simd NAME neon SOURCES simd/vec_test.cc simd/native_test.cc simd/vec_partial_boundary_test.cc
                simd/vec_half_combine_oracle_test.cc
                DEFINES HYLE_SIMD_TARGET_NEON OPTIONS ${_neon_flags})
  # SVE variant: the FIRST scalable-width backend (backend/sve.h). Its op face is VL-agnostic, so
  # it uses a DEDICATED sve_test.cc (not vec_test.cc, which pins Lanes to a compile-time constant).
  # Gated by HYLE_SIMD_SVE=ON because SVE needs `-march=...+sve` and a CPU/emulator that supports
  # it: on a real SVE host (Graviton3, etc.) ctest runs it directly; on a non-SVE dev box it is
  # cross-compiled (aarch64-linux-gnu-g++ -march=armv8-a+sve) and run under qemu-aarch64-static at
  # multiple VLs (128/256/512) -- the "one binary, every width" scalable proof. See the recipe in
  # docs/architecture/simd-design.md (SVE section) / scripts. NOT auto-built on a plain aarch64
  # build (would need +sve which some arm64 hosts lack), so it is opt-in.
  if(HYLE_SIMD_SVE)
    hyle_add_test(simd NAME sve SOURCES simd/sve_test.cc simd/native_test.cc
                  simd/scalable_math_compile.cc simd/scalable_math_product_test.cc
                  simd/scalable_math_dispatch_test.cc
                  DEFINES HYLE_SIMD_TARGET_SVE HYLE_SIMD_HAS_SVE_VARIANT=1
                  OPTIONS -march=armv8-a+sve)
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|riscv")
  # RVV variant (backend/rvv.h), the SECOND scalable backend. Same gating rationale as SVE:
  # RVV needs -march=rv64gcv + a CPU/emulator with the V extension. On a real RVV host ctest runs
  # it; on a non-RVV box it is cross-compiled (riscv64-linux-gnu-g++ -march=rv64gcv) and run under
  # qemu-riscv64-static (scripts/rvv-qemu-test.sh). Gated by HYLE_SIMD_RVV.
  if(HYLE_SIMD_RVV)
    hyle_add_test(simd NAME rvv SOURCES simd/rvv_test.cc simd/native_test.cc
                  simd/scalable_math_compile.cc simd/scalable_math_product_test.cc
                  simd/scalable_math_dispatch_test.cc
                  DEFINES HYLE_SIMD_TARGET_RVV HYLE_SIMD_HAS_RVV_VARIANT=1
                  OPTIONS -march=rv64gcv)
  endif()
endif()
if(HYLE_HAS_FCONTEXT)
  hyle_add_test(sched SOURCES sched/fiber_test.cc sched/run_queue_test.cc sched/scheduler_test.cc sched/waker_test.cc sched/sync_test.cc sched/timer_test.cc sched/coro_test.cc sched/affinity_test.cc sched/join_test.cc sched/blocking_test.cc sched/shutdown_test.cc sched/cancel_test.cc DEPS hyle_time)
  # graph-01 step-2: parallel evaluate over the sched work-stealing pool (level-by-level
  # fork-join). Needs sched (spawn_task/when_all/fiber). Same gate as sched itself.
  hyle_add_test(graph NAME parallel SOURCES graph/parallel_test.cc DEPS hyle_sched)
  # graph-02 step-1: data-parallel layer (parallel_for/parallel_reduce, recursive-split
  # fork-join over sched). Same sched gate.
  hyle_add_test(graph NAME data_parallel SOURCES graph/data_parallel_test.cc DEPS hyle_sched)
  # compute-03: co_await GPU stream completion -> fiber wake. Needs sched (the poller
  # wakes a fiber via Waker) -- only where fcontext exists, like the io co_await tests.
  hyle_add_test(compute NAME gpu_await SOURCES compute/gpu_await_test.cc DEPS hyle_sched hyle_mem)
  # compute-05: ggml_bridge -- borrow ggml operators on HYLE VRAM + spawn_blocking
  # completion + fiber scheduling. Needs sched (spawn_blocking/JoinHandle) AND ggml
  # fetched (HYLE_COMPUTE_GGML). Same gate as src/compute's ggml_bridge.cc.
  if(HYLE_COMPUTE_GGML)
    hyle_add_test(compute NAME ggml_bridge SOURCES compute/ggml_bridge_test.cc DEPS hyle_sched hyle_mem)
  endif()
  # io-04 net: the co_await socket layer needs BOTH io (proactor + IoContext) and
  # sched (scheduler drives the reap). Only buildable where sched exists.
  hyle_add_test(io NAME net SOURCES io/net_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME tcp SOURCES io/tcp_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME udp SOURCES io/udp_test.cc DEPS hyle_sched)
  # io-05 file io + io-09 async DNS / signal_set: co_await file ops, getaddrinfo
  # offload to the blocking pool, and signal waits -- all need io + sched.
  hyle_add_test(io NAME file SOURCES io/file_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME dns SOURCES io/dns_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME signal SOURCES io/signal_test.cc DEPS hyle_sched)
  hyle_add_test(io NAME timeout SOURCES io/timeout_test.cc DEPS hyle_sched hyle_time)
  # io-10 HTTP/1.1: server+client integration over loopback TCP (borrowed parser).
  hyle_add_test(io NAME http SOURCES io/http_test.cc DEPS hyle_sched)
  # sched-07 cancellable io: token-bound recv/accept, submit_cancel on cancel.
  hyle_add_test(io NAME net_cancel SOURCES io/net_cancel_test.cc DEPS hyle_sched hyle_time)
  # sched-07 consumer: HTTP request fan-out connected-cancel (request -> subtask tree).
  hyle_add_test(io NAME http_cancel SOURCES io/http_cancel_test.cc DEPS hyle_sched)
  # io-08 TLS: only when OpenSSL is available (matches the io module's optional TLS).
  # The test itself calls SSL_CTX_set_verify (client skips verify for the self-signed
  # cert), so it links OpenSSL directly too.
  if(OpenSSL_FOUND)
    hyle_add_test(io NAME tls SOURCES io/tls_test.cc DEPS hyle_sched)
    target_link_libraries(hyle_io_tls_test PRIVATE OpenSSL::SSL OpenSSL::Crypto)
    if(WIN32)
      target_link_libraries(hyle_io_tls_test PRIVATE ws2_32)
    endif()
  endif()
  if(WIN32)
    target_link_libraries(hyle_io_net_test PRIVATE ws2_32)  # test builds the loopback pair
    target_link_libraries(hyle_io_net_cancel_test PRIVATE ws2_32)  # loopback TCP pair
    target_link_libraries(hyle_io_tcp_test PRIVATE ws2_32)  # sockaddr / winsock types
    target_link_libraries(hyle_io_udp_test PRIVATE ws2_32)  # sockaddr / winsock types
    target_link_libraries(hyle_io_dns_test PRIVATE ws2_32)  # getaddrinfo + WSAStartup
    target_link_libraries(hyle_io_timeout_test PRIVATE ws2_32)  # loopback TCP pair
  endif()
endif()
