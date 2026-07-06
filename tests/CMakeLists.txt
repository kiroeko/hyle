include(FetchContent)

# GoogleTest (charter: unit test framework). Pinned tag.
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.15.2)
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
hyle_add_test(compute SOURCES compute/device_test.cc compute/buffer_test.cc)
hyle_add_test(container SOURCES container/bounded_queue_test.cc container/sharded_queue_test.cc container/work_stealing_deque_test.cc container/ws_fifo_queue_test.cc container/hash_test.cc container/carray_test.cc container/small_vector_test.cc container/probe_test.cc container/swiss_table_test.cc container/string_ref_test.cc container/string_buf_test.cc container/concurrent_vector_test.cc container/string_algo_test.cc container/path_test.cc DEPS hyle_platform mimalloc_static)
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
hyle_add_test(simd SOURCES simd/dispatch_test.cc simd/vec_test.cc simd/ops_test.cc simd/dtype_test.cc simd/math_test.cc)
# vec_test compiled per ISA variant: same source, different HYLE_SIMD_TARGET_* + the
# matching codegen flags from the single-source hyle_isa_flags (no MSVC/else drift here).
# Arch-gated (mirrors src/simd's CMAKE_SYSTEM_PROCESSOR split): x86 builds the sse4/avx2
# variants, aarch64 builds the neon variant -- feeding -mavx2 to an ARM build (or the ARM
# neon.h to an x86 build) would fail. Each variant runs on the build host, so it must be
# the host's real ISA (the variant-DLL test model).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_isa_flags(_sse4_flags SSE42)
  hyle_isa_flags(_avx2_flags AVX2)
  hyle_add_test(simd NAME sse4 SOURCES simd/vec_test.cc DEFINES HYLE_SIMD_TARGET_SSE4 OPTIONS ${_sse4_flags})
  hyle_add_test(simd NAME avx2 SOURCES simd/vec_test.cc DEFINES HYLE_SIMD_TARGET_AVX2 OPTIONS ${_avx2_flags})
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_isa_flags(_neon_flags NEON)
  hyle_add_test(simd NAME neon SOURCES simd/vec_test.cc DEFINES HYLE_SIMD_TARGET_NEON OPTIONS ${_neon_flags})
endif()
if(HYLE_HAS_FCONTEXT)
  hyle_add_test(sched SOURCES sched/fiber_test.cc sched/run_queue_test.cc sched/scheduler_test.cc sched/waker_test.cc sched/sync_test.cc sched/timer_test.cc sched/coro_test.cc sched/affinity_test.cc sched/join_test.cc sched/blocking_test.cc sched/shutdown_test.cc sched/cancel_test.cc DEPS hyle_time)
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
