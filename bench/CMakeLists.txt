# bench: microbenchmarks (charter: nanobench). Built only with -DHYLE_BUILD_BENCH=ON.
# nanobench is a single-header lib (vendored third_party/nanobench); compile its
# impl into one static lib that bench exes link. hyle_add_bench(<mod>) (see
# HyleModule.cmake) makes hyle_<mod>_bench linked to hyle_<mod> + nanobench.

add_library(nanobench STATIC nanobench_impl.cc)
target_include_directories(nanobench SYSTEM PUBLIC
  "${PROJECT_SOURCE_DIR}/third_party/nanobench")
find_package(Threads REQUIRED)
target_link_libraries(nanobench PUBLIC Threads::Threads)

hyle_add_bench(container SOURCES container/container_bench_main.cc container/swiss_table_bench.cc container/string_buf_bench.cc DEPS hyle_platform)
# mem bench links mimalloc_static directly so it can call raw mi_malloc/mi_free as
# the SOTA baseline (the static lib's PUBLIC include exposes mimalloc.h). hyle_mem
# links mimalloc PRIVATE, so the bench can't borrow the symbol transitively.
hyle_add_bench(mem SOURCES mem/allocator_bench.cc DEPS mimalloc_static)

# arena/pool/stack specialized-allocator cluster vs std::pmr (std's arena/pool, no
# external clone) + raw mimalloc. Own exe (allocator_bench.cc already owns main()).
add_executable(hyle_arena_pool_bench mem/arena_pool_bench.cc)
target_link_libraries(hyle_arena_pool_bench PRIVATE hyle_mem nanobench mimalloc_static)
hyle_apply_warnings(hyle_arena_pool_bench)

# sched bench: fiber switch (~23ns) + spawn throughput vs std::async (~7-9x). Now
# compiled into the default bench build (no longer withheld). It once crashed at
# teardown with mimalloc ON, but an exhaustive hunt (docs/debug/ki-1-crash-hunt.md)
# RULED OUT a HYLE logic bug -- TSan (no source race) + MI_DEBUG=3 full heap checks
# (no corruption) are clean under churn + oversubscription + 24G memory pressure,
# and the original crasher is NOT reproducible on native ext4. Evidence (SIGILL +
# clean instrumented runs) points to a 9p/WSL paging artifact, not this code. Kept
# IN the build on purpose: if it ever recurs, we investigate head-on, not hide it.
hyle_add_bench(sched SOURCES sched/scheduler_bench.cc)

# sched-05 fiber-aware sync primitives bench (own exe: scheduler_bench.cc already
# owns main(), so this is a second target rather than another source in it).
add_executable(hyle_sched_sync_bench sched/sync_bench.cc)
target_link_libraries(hyle_sched_sync_bench PRIVATE hyle_sched nanobench)
hyle_apply_warnings(hyle_sched_sync_bench)

# task #9: lock-free queue cluster SOTA parity. HYLE baselines (BoundedQueue,
# WsFifoQueue, WorkStealingDeque, ConcurrentVector) ALWAYS compile into this exe;
# the SOTA rivals below are opt-in (each is a heavy/header dep we don't force).
# The same-family work-stealing rival (crossbeam-deque) is a SEPARATE Rust crate
# (bench/rust/crossbeam_queue_bench) since it can't link a C++ TU -- run it on its
# own and lay the ns/op side by side (see docs/architecture/performance.md).
add_executable(hyle_queue_sota_bench container/queue_sota_bench.cc)
target_link_libraries(hyle_queue_sota_bench PRIVATE hyle_container nanobench)
hyle_apply_warnings(hyle_queue_sota_bench)

# task #19 breadth: the two claim-bearing containers with no prior bench. SmallVector
# (inline storage, "省 malloc") vs std::vector; IntrusiveList ("O(1) 无分配") vs
# std::list. SmallVector needs system_allocator -> link hyle_mem too.
add_executable(hyle_container_misc_bench container/container_misc_bench.cc)
target_link_libraries(hyle_container_misc_bench PRIVATE hyle_container hyle_mem nanobench)
hyle_apply_warnings(hyle_container_misc_bench)

# Optional: moodycamel::ConcurrentQueue (header-only -> just an include path).
# Enable with -DHYLE_BENCH_MOODYCAMEL=ON; point HYLE_MOODYCAMEL_DIR at the cloned
# repo (sibling of hyle/ by default).
option(HYLE_BENCH_MOODYCAMEL "build queue bench vs moodycamel ConcurrentQueue" OFF)
if(HYLE_BENCH_MOODYCAMEL)
  set(HYLE_MOODYCAMEL_DIR "${PROJECT_SOURCE_DIR}/../moodycamel-ConcurrentQueue"
      CACHE PATH "cloned moodycamel ConcurrentQueue")
  if(NOT EXISTS "${HYLE_MOODYCAMEL_DIR}/concurrentqueue.h")
    message(FATAL_ERROR "HYLE_BENCH_MOODYCAMEL=ON but no concurrentqueue.h at ${HYLE_MOODYCAMEL_DIR}")
  endif()
  target_compile_definitions(hyle_queue_sota_bench PRIVATE HYLE_BENCH_MOODYCAMEL)
  target_include_directories(hyle_queue_sota_bench SYSTEM PRIVATE "${HYLE_MOODYCAMEL_DIR}")
endif()

# Optional: oneTBB concurrent_bounded_queue + concurrent_vector (the apples-to-
# apples bounded-MPMC and concurrent-vector rivals; same try_push/try_pop/push_back
# API). Heavy external build -> add_subdirectory the cloned oneTBB. Enable with
# -DHYLE_BENCH_TBB=ON; point HYLE_TBB_DIR at the clone (sibling by default).
option(HYLE_BENCH_TBB "build queue/vector bench vs oneTBB" OFF)
if(HYLE_BENCH_TBB)
  set(HYLE_TBB_DIR "${PROJECT_SOURCE_DIR}/../oneTBB" CACHE PATH "cloned oneTBB")
  if(NOT EXISTS "${HYLE_TBB_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_TBB=ON but no oneTBB at ${HYLE_TBB_DIR}")
  endif()
  set(TBB_TEST OFF CACHE BOOL "" FORCE)
  set(TBBMALLOC_BUILD OFF CACHE BOOL "" FORCE)
  set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory("${HYLE_TBB_DIR}" "${CMAKE_BINARY_DIR}/onetbb" EXCLUDE_FROM_ALL)
  target_compile_definitions(hyle_queue_sota_bench PRIVATE HYLE_BENCH_TBB)
  target_link_libraries(hyle_queue_sota_bench PRIVATE TBB::tbb)
endif()

# simd bench: dot/saxpy raced across scalar-autovec / HYLE sse4 / HYLE avx2 (+ Highway if
# enabled). Each kernel TU is compiled with ITS OWN -march so the per-ISA codegen is real
# and isolated (one backend per TU). The scalar baseline TU gets NO -march (auto-vec only
# = the honest "free from the optimizer" reference). Built into one exe.
add_executable(hyle_simd_bench
  simd/simd_bench.cc
  simd/simd_kernels_scalar.cc
  simd/simd_kernels_sse4.cc
  simd/simd_kernels_avx2.cc
  simd/simd_kernels_highway.cc)
target_link_libraries(hyle_simd_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_simd_bench PRIVATE "${PROJECT_SOURCE_DIR}")
if(NOT MSVC)
  set_source_files_properties(simd/simd_kernels_sse4.cc PROPERTIES COMPILE_OPTIONS "-msse4.2")
  set_source_files_properties(simd/simd_kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
else()
  # MSVC x64 has SSE4.2 intrinsics unconditionally; AVX2 needs /arch:AVX2. scalar TU gets
  # neither (stays SSE2 auto-vec baseline).
  set_source_files_properties(simd/simd_kernels_avx2.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
endif()
hyle_apply_warnings(hyle_simd_bench)

# math bench: vectorized exp/log/tanh (SLEEF AVX2 when hyle_simd built with
# -DHYLE_SIMD_SLEEF=ON) vs scalar libm. Own exe (simd_bench.cc owns main()). The
# *_f32_batch ops live in hyle_simd; SLEEF is its PRIVATE dep, so this just links
# hyle_simd. Without SLEEF the HYLE row ties the scalar baseline (honest no-vec).
add_executable(hyle_math_bench simd/math_bench.cc)
target_link_libraries(hyle_math_bench PRIVATE hyle_simd nanobench)
hyle_apply_warnings(hyle_math_bench)

# Optional SOTA parity: HYLE avx2 vs Highway avx2, head to head (same ISA). Enable with
# -DHYLE_BENCH_HIGHWAY=ON; point HYLE_HIGHWAY_DIR at the cloned highway (sibling default).
option(HYLE_BENCH_HIGHWAY "build the simd-vs-Highway parity bench" OFF)
if(HYLE_BENCH_HIGHWAY)
  set(HYLE_HIGHWAY_DIR "${PROJECT_SOURCE_DIR}/../highway" CACHE PATH "cloned highway")
  if(NOT EXISTS "${HYLE_HIGHWAY_DIR}/hwy/highway.h")
    message(FATAL_ERROR "HYLE_BENCH_HIGHWAY=ON but no highway at ${HYLE_HIGHWAY_DIR}")
  endif()
  target_compile_definitions(hyle_simd_bench PRIVATE HYLE_BENCH_HIGHWAY)
  target_include_directories(hyle_simd_bench SYSTEM PRIVATE "${HYLE_HIGHWAY_DIR}")
  # Highway static-dispatch ops are header-only; the kernel TU is built at -mavx2 so its
  # HWY_STATIC_TARGET resolves to AVX2 (parity with HYLE avx2).
  if(NOT MSVC)
    set_source_files_properties(simd/simd_kernels_highway.cc PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-mf16c")
  else()
    set_source_files_properties(simd/simd_kernels_highway.cc PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
  endif()
endif()

# Optional SOTA parity: HYLE fiber sync vs marl (the fiber-sync library this
# design converged from). Off by default (marl is an external build). Enable with
# -DHYLE_BENCH_MARL=ON; point HYLE_MARL_DIR at a cloned marl (sibling default).
option(HYLE_BENCH_MARL "build the fiber-sync-vs-marl parity bench" OFF)
if(HYLE_BENCH_MARL)
  set(HYLE_MARL_DIR "${PROJECT_SOURCE_DIR}/../marl" CACHE PATH "cloned marl")
  if(NOT EXISTS "${HYLE_MARL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_MARL=ON but no marl at ${HYLE_MARL_DIR}")
  endif()
  add_subdirectory("${HYLE_MARL_DIR}" "${CMAKE_BINARY_DIR}/marl" EXCLUDE_FROM_ALL)
  add_executable(hyle_sched_marl_bench sched/sync_marl_bench.cc)
  target_link_libraries(hyle_sched_marl_bench PRIVATE hyle_sched nanobench marl)
  hyle_apply_warnings(hyle_sched_marl_bench)
endif()

# Optional SOTA parity bench: SwissTable vs absl::flat_hash_map. Off by default
# (abseil is a heavy external build). Enable with -DHYLE_BENCH_ABSL=ON; point
# HYLE_ABSL_DIR at a cloned abseil-cpp (defaults to a sibling of the repo).
option(HYLE_BENCH_ABSL "build the SwissTable-vs-absl parity bench" OFF)
if(HYLE_BENCH_ABSL)
  set(HYLE_ABSL_DIR "${PROJECT_SOURCE_DIR}/../abseil-cpp" CACHE PATH "cloned abseil-cpp")
  if(NOT EXISTS "${HYLE_ABSL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_ABSL=ON but no abseil at ${HYLE_ABSL_DIR}")
  endif()
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_ENABLE_INSTALL OFF)
  add_subdirectory("${HYLE_ABSL_DIR}" "${CMAKE_BINARY_DIR}/abseil" EXCLUDE_FROM_ALL)
  add_executable(hyle_absl_bench container/swiss_vs_absl_bench.cc)
  target_link_libraries(hyle_absl_bench PRIVATE hyle_container nanobench
    absl::flat_hash_map)
  hyle_apply_warnings(hyle_absl_bench)
endif()
