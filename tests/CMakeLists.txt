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
# close-01 sample: pure C TU (C compiler) linking only the core C ABI header.
add_executable(hyle_core_c_test core/c_core_test.c)
target_link_libraries(hyle_core_c_test PRIVATE hyle_core)
set_source_files_properties(core/c_core_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_core_c_test)
hyle_apply_sanitizer(hyle_core_c_test)
add_test(NAME hyle_core_c_test COMMAND hyle_core_c_test)
hyle_add_test(error SOURCES error/status_test.cc error/error_handler_test.cc)
# close-01 sample: pure C TU (C compiler) linking only the C ABI header.
add_executable(hyle_error_c_test error/c_error_test.c)
target_link_libraries(hyle_error_c_test PRIVATE hyle_error)
# Ensure the .c file is compiled as C, not C++ (some generators sniff extension).
set_source_files_properties(error/c_error_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_error_c_test)
hyle_apply_sanitizer(hyle_error_c_test)
add_test(NAME hyle_error_c_test COMMAND hyle_error_c_test)

add_executable(hyle_platform_c_test platform/c_platform_test.c)
target_link_libraries(hyle_platform_c_test PRIVATE hyle_platform hyle_error)
set_source_files_properties(platform/c_platform_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_platform_c_test)
hyle_apply_sanitizer(hyle_platform_c_test)
add_test(NAME hyle_platform_c_test COMMAND hyle_platform_c_test)

# close-01: pure C TU linking only the mem C ABI header (hyle/c/mem.h -> hyle_mem).
add_executable(hyle_mem_c_test mem/c_mem_test.c)
target_link_libraries(hyle_mem_c_test PRIVATE hyle_mem)
set_source_files_properties(mem/c_mem_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_mem_c_test)
hyle_apply_sanitizer(hyle_mem_c_test)
add_test(NAME hyle_mem_c_test COMMAND hyle_mem_c_test)

# close-01: pure C whole-buffer SIMD consumer. Register/native C++ types never cross this edge.
add_executable(hyle_simd_c_test simd/c_simd_test.c)
target_link_libraries(hyle_simd_c_test PRIVATE hyle_simd)
set_source_files_properties(simd/c_simd_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_simd_c_test)
hyle_apply_sanitizer(hyle_simd_c_test)
add_test(NAME hyle_simd_c_test COMMAND hyle_simd_c_test)

hyle_add_test(obs SOURCES obs/log_test.cc)
add_executable(hyle_obs_c_test obs/c_obs_test.c)
target_link_libraries(hyle_obs_c_test PRIVATE hyle_obs)
set_source_files_properties(obs/c_obs_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_obs_c_test)
hyle_apply_sanitizer(hyle_obs_c_test)
add_test(NAME hyle_obs_c_test COMMAND hyle_obs_c_test)
hyle_add_test(platform SOURCES platform/wait_test.cc platform/thread_test.cc platform/dynlib_test.cc platform/fs_test.cc platform/virtual_memory_test.cc platform/numa_test.cc platform/mapped_file_test.cc platform/cpu_test.cc)
hyle_add_test(mem SOURCES mem/allocator_test.cc mem/shared_pool_test.cc mem/pool_stack_test.cc mem/alignment_test.cc mem/pmr_test.cc mem/growable_pool_test.cc mem/fiber_stack_test.cc mem/stats_test.cc mem/hugepage_test.cc mem/numa_test.cc DEPS hyle_platform mimalloc_static)
hyle_add_test(util SOURCES util/ref_counted_test.cc util/scope_guard_test.cc util/type_name_test.cc util/registry_test.cc util/random_test.cc util/spin_lock_test.cc util/demangle_test.cc util/c_util_concurrency_test.cc DEPS hyle_container)
add_executable(hyle_util_c_test util/c_util_test.c)
target_link_libraries(hyle_util_c_test PRIVATE hyle_util)
set_source_files_properties(util/c_util_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_util_c_test)
hyle_apply_sanitizer(hyle_util_c_test)
add_test(NAME hyle_util_c_test COMMAND hyle_util_c_test)
hyle_add_test(time SOURCES time/header_test.cc time/clock_test.cc
  time/timer_wheel_test.cc time/common_timeout_queue_test.cc DEPS hyle_platform)
add_executable(hyle_time_c_test time/c_time_test.c)
target_link_libraries(hyle_time_c_test PRIVATE hyle_time)
set_source_files_properties(time/c_time_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_time_c_test)
hyle_apply_sanitizer(hyle_time_c_test)
add_test(NAME hyle_time_c_test COMMAND hyle_time_c_test)
hyle_add_test(io SOURCES io/proactor_test.cc)
add_executable(hyle_io_c_test io/c_io_test.c)
target_link_libraries(hyle_io_c_test PRIVATE hyle_io hyle_mem)
set_source_files_properties(io/c_io_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_io_c_test)
hyle_apply_sanitizer(hyle_io_c_test)
add_test(NAME hyle_io_c_test COMMAND hyle_io_c_test)
if(WIN32)
  # The io test itself calls Winsock directly (WSASocketW/bind/getsockname/... in
  # the IOCP TCP round-trip test), so it needs ws2_32 linked into the test exe.
  target_link_libraries(hyle_io_test PRIVATE ws2_32)
endif()
hyle_add_test(math SOURCES math/scalar_test.cc)
# compute-01: device/stream/event + completion. Always buildable (host fallback);
# the GPU-specific test self-skips at runtime when no CUDA device is present. The
# test TU is pure C++ (all CUDA is inside hyle_compute), so it needs no nvcc.
# compute-02: owned backend buffer + offset copy + explicit host mapping/coherence.
# Headline tests = KV-cache repeated offset writes and direct HostBuffer mapping.
hyle_add_test(compute SOURCES compute/device_test.cc compute/buffer_test.cc compute/device_pool_test.cc compute/kernel_test.cc compute/contract_test.cc compute/host_ops_test.cc)
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
  # Best-effort non-coherent HOST_CACHED variant. The test queries the selected memory type and
  # skips honestly when the physical device exposes only coherent host-visible memory.
  add_test(NAME hyle_compute_vulkan_noncoherent_test
           COMMAND hyle_compute_vulkan_test
                   --gtest_filter=VulkanCompute.NonCoherentHostBufferRoundTrip)
  set_tests_properties(hyle_compute_vulkan_noncoherent_test PROPERTIES
                       ENVIRONMENT "HYLE_VK_FORCE_NONCOHERENT=1")
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
hyle_add_test(container SOURCES container/bounded_queue_test.cc container/sharded_queue_test.cc container/work_stealing_deque_test.cc container/ws_fifo_queue_test.cc container/hash_test.cc container/carray_test.cc container/c_container_concurrency_test.cc container/small_vector_test.cc container/probe_test.cc container/swiss_table_test.cc container/string_ref_test.cc container/string_buf_test.cc container/concurrent_vector_test.cc container/string_algo_test.cc container/path_test.cc DEPS hyle_platform mimalloc_static)
add_executable(hyle_container_c_test container/c_container_test.c)
target_link_libraries(hyle_container_c_test PRIVATE hyle_container)
set_source_files_properties(container/c_container_test.c PROPERTIES LANGUAGE C)
hyle_apply_warnings(hyle_container_c_test)
hyle_apply_sanitizer(hyle_container_c_test)
add_test(NAME hyle_container_c_test COMMAND hyle_container_c_test)
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
target_include_directories(hyle_simd_test PRIVATE "${PROJECT_SOURCE_DIR}")
hyle_add_fat_kernel(
  TARGET hyle_simd_test
  NAME public_fat_fixture
  IMPL "tests/simd/fat_kernel_fixture.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
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
  hyle_add_test(simd NAME avx2 SOURCES simd/vec_test.cc simd/native_test.cc
                simd/sort_kernel_test.cc ${_fixed_partial_tests}
                DEFINES HYLE_SIMD_TARGET_AVX2 OPTIONS ${_avx2_flags})
  target_include_directories(hyle_simd_avx2_test PRIVATE "${PROJECT_SOURCE_DIR}")
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
  # Default: build for compile-validation only, do NOT auto-run (bare metal
  # SIGILLs on a non-AVX512 host, e.g. WSL2/Hyper-V hides AVX-512). Set
  # HYLE_SIMD_AVX512_RUNNABLE (a真 AVX-512 host, or ctest wrapped in `sde64
  # -future`) to register avx512/avx512_dl as auto-run ctests.
  if(HYLE_SIMD_AVX512_RUNNABLE)
    set(_avx512_autorun "")
  else()
    set(_avx512_autorun NO_AUTORUN)
  endif()
  hyle_add_test(simd NAME avx512 SOURCES simd/vec_test.cc simd/native_test.cc
                simd/sort_kernel_test.cc ${_fixed_partial_tests}
                DEFINES HYLE_SIMD_TARGET_AVX512 OPTIONS ${_avx512_flags}
                ${_avx512_autorun})
  target_include_directories(hyle_simd_avx512_test PRIVATE "${PROJECT_SOURCE_DIR}")
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
    hyle_add_test(simd NAME avx512_dl SOURCES simd/vec_test.cc simd/native_test.cc DEFINES HYLE_SIMD_TARGET_AVX512 OPTIONS ${_avx512dl_flags} ${_avx512_autorun})
  endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_isa_flags(_neon_flags NEON)
  # The same fixed-width partial proof covers NEON's physical D/Q split. All cases
  # are permanent default-on regressions.
  hyle_add_test(simd NAME neon SOURCES simd/vec_test.cc simd/native_test.cc
                simd/sort_kernel_test.cc simd/vec_partial_boundary_test.cc
                simd/vec_half_combine_oracle_test.cc
                DEFINES HYLE_SIMD_TARGET_NEON OPTIONS ${_neon_flags})
  target_include_directories(hyle_simd_neon_test PRIVATE "${PROJECT_SOURCE_DIR}")
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
                  simd/scalable_math_dispatch_test.cc simd/scalable_dtype_product_test.cc
                  simd/scalable_bitpack_product_test.cc simd/fixed128_bitpack_reference.cc
                  simd/scalable_sort_product_test.cc
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
                  simd/scalable_math_dispatch_test.cc simd/scalable_dtype_product_test.cc
                  simd/scalable_bitpack_product_test.cc simd/fixed128_bitpack_reference.cc
                  simd/scalable_sort_product_test.cc
                  DEFINES HYLE_SIMD_TARGET_RVV HYLE_SIMD_HAS_RVV_VARIANT=1
                  OPTIONS -march=rv64gcv)
  endif()
endif()
if(HYLE_HAS_FCONTEXT)
  hyle_add_test(sched SOURCES sched/fiber_test.cc sched/fiber_stack_guard_test.cc sched/run_queue_test.cc sched/scheduler_test.cc sched/waker_test.cc sched/sync_test.cc sched/timer_test.cc sched/coro_test.cc sched/affinity_test.cc sched/join_test.cc sched/blocking_test.cc sched/shutdown_test.cc sched/cancel_test.cc DEPS hyle_time)
  add_executable(hyle_sched_c_test sched/c_sched_test.c)
  target_link_libraries(hyle_sched_c_test PRIVATE hyle_sched)
  set_source_files_properties(sched/c_sched_test.c PROPERTIES LANGUAGE C)
  hyle_apply_warnings(hyle_sched_c_test)
  hyle_apply_sanitizer(hyle_sched_c_test)
  add_test(NAME hyle_sched_c_test COMMAND hyle_sched_c_test)
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
  if(HYLE_IO_HTTP2)
    # io-11: real h2c loopback with two interleaved streams and reverse-order
    # responses (HPACK + multiplex + flow-control session path).
    hyle_add_test(io NAME http2 SOURCES io/http2_test.cc DEPS hyle_sched)
  endif()
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
    if(HYLE_IO_HTTP2)
      target_link_libraries(hyle_io_http2_test PRIVATE ws2_32)
    endif()
  endif()
endif()
