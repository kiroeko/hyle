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
hyle_isa_flags(_sse4_flags SSE42)
hyle_isa_flags(_avx2_flags AVX2)
hyle_add_test(simd NAME sse4 SOURCES simd/vec_test.cc DEFINES HYLE_SIMD_TARGET_SSE4 OPTIONS ${_sse4_flags})
hyle_add_test(simd NAME avx2 SOURCES simd/vec_test.cc DEFINES HYLE_SIMD_TARGET_AVX2 OPTIONS ${_avx2_flags})
if(HYLE_HAS_FCONTEXT)
  hyle_add_test(sched SOURCES sched/fiber_test.cc sched/run_queue_test.cc sched/scheduler_test.cc sched/waker_test.cc sched/sync_test.cc sched/timer_test.cc sched/coro_test.cc DEPS hyle_time)
endif()
