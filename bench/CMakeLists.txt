# bench: microbenchmarks (charter: nanobench). Built only with -DHYLE_BUILD_BENCH=ON.
# nanobench is a single-header lib (vendored third_party/nanobench); compile its
# impl into one static lib that bench exes link. hyle_add_bench(<mod>) (see
# HyleModule.cmake) makes hyle_<mod>_bench linked to hyle_<mod> + nanobench.

add_library(nanobench STATIC nanobench_impl.cc)
target_include_directories(nanobench SYSTEM PUBLIC
  "${PROJECT_SOURCE_DIR}/third_party/nanobench")
find_package(Threads REQUIRED)
target_link_libraries(nanobench PUBLIC Threads::Threads)

hyle_add_bench(container SOURCES container/container_bench_main.cc container/swiss_table_bench.cc container/string_buf_bench.cc container/string_ref_bench.cc container/path_bench.cc container/string_algo_bench.cc DEPS hyle_platform)
# string_algo utf8 bench races simdutf (the SIMD utf8 SOTA) for the comparison row.
# simdutf is VENDORED in third_party and compiled into hyle_container (the product uses it
# unconditionally), but those symbols are PRIVATE/non-exported across the hyle_container DLL
# boundary -- the bench can't borrow them transitively. So compile simdutf.cpp into the bench
# exe too: it's a SEPARATE binary from the DLL, so its own copy is NOT a duplicate symbol (the
# DLL exports none of them). The gate just toggles the bench-side comparison row.
option(HYLE_BENCH_SIMDUTF "build string_algo utf8 bench vs simdutf (SIMD SOTA)" OFF)
if(HYLE_BENCH_SIMDUTF)
  set(_simdutf "${PROJECT_SOURCE_DIR}/third_party/simdutf")
  target_compile_definitions(hyle_container_bench PRIVATE HYLE_BENCH_SIMDUTF)
  target_include_directories(hyle_container_bench SYSTEM PRIVATE "${_simdutf}")
  target_sources(hyle_container_bench PRIVATE "${_simdutf}/simdutf.cpp")
  # Same -Wall/-Werror exemption as the product build (src/container): the amalgamation's
  # own -Wpedantic warnings would break a HYLE_WERROR=ON bench build.
  if(MSVC)
    set_source_files_properties("${_simdutf}/simdutf.cpp" PROPERTIES COMPILE_OPTIONS "/w")
  else()
    set_source_files_properties("${_simdutf}/simdutf.cpp" PROPERTIES COMPILE_OPTIONS "-w")
  endif()
endif()
# mem bench links mimalloc_static directly so it can call raw mi_malloc/mi_free as
# the SOTA baseline (the static lib's PUBLIC include exposes mimalloc.h). hyle_mem
# links mimalloc PRIVATE, so the bench can't borrow the symbol transitively.
hyle_add_bench(mem SOURCES mem/allocator_bench.cc DEPS mimalloc_static)

# arena/pool/stack specialized-allocator cluster vs std::pmr (std's arena/pool, no
# external clone) + raw mimalloc. Own exe (allocator_bench.cc already owns main()).
add_executable(hyle_arena_pool_bench mem/arena_pool_bench.cc)
target_link_libraries(hyle_arena_pool_bench PRIVATE hyle_mem nanobench mimalloc_static)
hyle_apply_warnings(hyle_arena_pool_bench)

# platform primitive bench (red-team gap-fill): Parker park/unpark (futex/
# WaitOnAddress) + Thread spawn/join. Links hyle_platform (Parker/Thread live there).
add_executable(hyle_platform_bench platform/platform_bench.cc)
target_link_libraries(hyle_platform_bench PRIVATE hyle_platform nanobench)
hyle_apply_warnings(hyle_platform_bench)

# sched bench: fiber switch (~23ns) + spawn throughput vs std::async (~7-9x). Now
# compiled into the default bench build (no longer withheld). It once crashed at
# teardown with mimalloc ON, but an exhaustive hunt (docs/debug/ki-1-crash-hunt.md)
# RULED OUT a HYLE logic bug -- TSan (no source race) + MI_DEBUG=3 full heap checks
# (no corruption) are clean under churn + oversubscription + 24G memory pressure,
# and the original crasher is NOT reproducible on native ext4. Evidence (SIGILL +
# clean instrumented runs) points to a 9p/WSL paging artifact, not this code. Kept
# IN the build on purpose: if it ever recurs, we investigate head-on, not hide it.
hyle_add_bench(sched SOURCES sched/scheduler_bench.cc)

# io echo bench (io-04): the concurrent-connection perf comparison the io-01/02/03
# notes deferred until there was a real load. Drives kConns x kRoundTrips echoes
# through HYLE's shard-per-core stack at 1/2/4/8 workers (scaling) + a raw epoll
# readiness-loop baseline (Linux). Hard-depends on sched (io<->sched integration),
# so gate on fcontext like sched itself. Needs ws2_32 on Windows (socketpair
# emulation via loopback TCP).
if(HYLE_HAS_FCONTEXT)
  hyle_add_bench(io SOURCES io/echo_bench.cc DEPS hyle_sched)
  if(WIN32)
    target_link_libraries(hyle_io_bench PRIVATE ws2_32)
  endif()

  # Optional SOTA parity: echo_bench's HYLE rows vs a hand-rolled raw-liburing echo on
  # the SAME 64-conn/64B ping-pong load (Linux only -- liburing is io_uring). This is
  # the head-to-head "does HYLE's co_await + shard-per-core add overhead over a bare
  # io_uring loop written with the community-standard wrapper?" comparison. Enable with
  # -DHYLE_BENCH_LIBURING=ON; point HYLE_LIBURING_DIR at a cloned liburing (sibling
  # default). liburing uses a Makefile (not CMake), so we configure+make its static lib
  # once at configure time and link the resulting src/liburing.a + its headers.
  option(HYLE_BENCH_LIBURING "build echo bench vs raw liburing (io_uring SOTA)" OFF)
  if(HYLE_BENCH_LIBURING AND UNIX AND NOT APPLE)
    set(HYLE_LIBURING_DIR "${PROJECT_SOURCE_DIR}/../liburing" CACHE PATH "cloned liburing")
    if(NOT EXISTS "${HYLE_LIBURING_DIR}/src/include/liburing.h")
      message(FATAL_ERROR "HYLE_BENCH_LIBURING=ON but no liburing at ${HYLE_LIBURING_DIR}")
    endif()
    if(NOT EXISTS "${HYLE_LIBURING_DIR}/src/liburing.a")
      message(STATUS "liburing: configure + make static lib (one-time)")
      execute_process(COMMAND ./configure WORKING_DIRECTORY "${HYLE_LIBURING_DIR}"
        RESULT_VARIABLE _lu_cfg OUTPUT_QUIET ERROR_QUIET)
      execute_process(COMMAND make -j -C src WORKING_DIRECTORY "${HYLE_LIBURING_DIR}"
        RESULT_VARIABLE _lu_make OUTPUT_QUIET ERROR_QUIET)
      if(_lu_make OR NOT EXISTS "${HYLE_LIBURING_DIR}/src/liburing.a")
        message(FATAL_ERROR "liburing build failed (configure=${_lu_cfg} make=${_lu_make}); run `./configure && make -C src` in ${HYLE_LIBURING_DIR}")
      endif()
    endif()
    target_compile_definitions(hyle_io_bench PRIVATE HYLE_BENCH_LIBURING)
    target_include_directories(hyle_io_bench SYSTEM PRIVATE "${HYLE_LIBURING_DIR}/src/include")
    target_link_libraries(hyle_io_bench PRIVATE "${HYLE_LIBURING_DIR}/src/liburing.a")
  endif()

  # io scaling bench: the two COMPLEXITY curves echo_bench's fixed 64-conn/64B load
  # can't show -- connection-count scaling (ns/round-trip flat => O(1)/conn, the
  # "proactor beats readiness" argument) + payload-size scaling (bytes/s clean plateau
  # => no hidden copy in the co_await path). Own exe (echo_bench owns main). Same sched
  # dep + ws2_32 on Windows (socketpair emulation via loopback TCP).
  add_executable(hyle_io_scaling_bench io/io_scaling_bench.cc)
  target_link_libraries(hyle_io_scaling_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_io_scaling_bench)
  if(WIN32)
    target_link_libraries(hyle_io_scaling_bench PRIVATE ws2_32)
  endif()

  # io accept-scaling bench: isolates CONNECTION ESTABLISHMENT (the accept path the
  # echo benches never time -- they pre-build sockets). One TcpListener accepts N
  # conns while N client fibers connect; reports accepts/s swept over N to show the
  # accept path is O(1)/connection. Real TCP over loopback (accept needs a listener),
  # so hyle_io (tcp wrappers) + ws2_32 on Windows.
  add_executable(hyle_accept_bench io/accept_bench.cc)
  target_link_libraries(hyle_accept_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_accept_bench)
  if(WIN32)
    target_link_libraries(hyle_accept_bench PRIVATE ws2_32)
  endif()
endif()

# error-module bench: Status/Result (zero-dependency root, every hot-path return type)
# vs std::expected + absl::Status/StatusOr (the same tagged-pointer design; gated
# -DHYLE_BENCH_ABSL, absl::status/statusor linked in the absl block below). Own exe.
add_executable(hyle_error_bench error/status_bench.cc)
target_link_libraries(hyle_error_bench PRIVATE hyle_error nanobench)
hyle_apply_warnings(hyle_error_bench)

# misc-module bench (gap-fill #37): util/random (xoshiro256++ vs std::mt19937_64),
# time/clock (monotonic/realtime vs std::chrono), obs/log (produce-path overhead).
# random is header-only (util); clock needs hyle_time, log needs hyle_obs.
add_executable(hyle_misc_bench misc/misc_bench.cc)
target_link_libraries(hyle_misc_bench PRIVATE hyle_time hyle_obs hyle_util nanobench)
hyle_apply_warnings(hyle_misc_bench)

# time-02 timer wheel bench: the hierarchical wheel vs a min-heap (Go/libev-style,
# lazy cancel) across the agent workload (arm+cancel / arm+expire / churn). The
# heap is the honest same-job ALGORITHM rival (folly HHWheelTimer is welded to an
# EventBase loop -> not an apples-to-apples data-structure race). Own exe.
add_executable(hyle_timer_wheel_bench time/timer_wheel_bench.cc)
target_link_libraries(hyle_timer_wheel_bench PRIVATE hyle_time nanobench)
hyle_apply_warnings(hyle_timer_wheel_bench)

# sched-05 fiber-aware sync primitives bench (own exe: scheduler_bench.cc already
# owns main(), so this is a second target rather than another source in it).
add_executable(hyle_sched_sync_bench sched/sync_bench.cc)
target_link_libraries(hyle_sched_sync_bench PRIVATE hyle_sched nanobench)
hyle_apply_warnings(hyle_sched_sync_bench)

# sched-10 JoinHandle/spawn_task/when_all/when_any bench: proves the structured-
# concurrency layer is ~zero-overhead vs the raw fiber+WaitGroup(1)+slot it wraps.
# Own exe (scheduler_bench owns main). Needs fcontext like the rest of sched.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_join_bench sched/join_bench.cc)
  target_link_libraries(hyle_sched_join_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_join_bench)
endif()

# scheduler-layer gap benches (work-stealing scaling, spawn_on affinity,
# coroutine glue vs raw fiber spine, fiber sleep). Complements scheduler_bench
# (raw primitives) + queue_sota_bench (the work-stealing DATA STRUCTURE) by
# measuring SCHEDULER-LEVEL behaviour. Own exe. Needs fcontext (coro + timer).
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_ext_bench sched/scheduler_ext_bench.cc)
  target_link_libraries(hyle_sched_ext_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_ext_bench)
endif()

# task #9: lock-free queue cluster SOTA parity. HYLE baselines (BoundedQueue,
# WsFifoQueue, WorkStealingDeque, ConcurrentVector) ALWAYS compile into this exe;
# the SOTA rivals below are opt-in (each is a heavy/header dep we don't force).
# The same-family work-stealing rival (crossbeam-deque) is a SEPARATE Rust crate
# (bench/rust/crossbeam_queue_bench) since it can't link a C++ TU -- run it on its
# own and lay the ns/op side by side (see docs/architecture/performance.md).
add_executable(hyle_queue_sota_bench container/queue_sota_bench.cc)
# BoundedQueue/ShardedQueue default to compile-time SystemAllocator -> inline mi_malloc
# into this exe; link mimalloc_static (symbols are internal to hyle_mem.dll).
target_link_libraries(hyle_queue_sota_bench PRIVATE hyle_container nanobench mimalloc_static)
hyle_apply_warnings(hyle_queue_sota_bench)

# task #19 breadth + gap-fill #34/#35: the claim-bearing containers. SmallVector
# (inline storage) vs std::vector; IntrusiveList ("O(1) 无分配") vs std::list;
# CArray (dynamic array) vs std::vector. Real SOTA rivals are opt-in: absl::
# InlinedVector (-DHYLE_BENCH_ABSL, wired in the absl block below) and boost::
# intrusive::list (-DHYLE_BENCH_BOOST, header-only block below). SmallVector
# defaults to the compile-time SystemAllocator which INLINES mi_malloc into THIS
# exe -> link mimalloc_static directly (symbols internal to hyle_mem.dll).
add_executable(hyle_container_misc_bench container/container_misc_bench.cc)
target_link_libraries(hyle_container_misc_bench PRIVATE hyle_container hyle_mem nanobench mimalloc_static)
hyle_apply_warnings(hyle_container_misc_bench)

# boost::intrusive::list -- the canonical intrusive-container SOTA (header-only).
# Enable with -DHYLE_BENCH_BOOST=ON; point HYLE_BOOST_DIR at a boost tree (the
# superproject root with boost/intrusive/list.hpp reachable, e.g. a full boost
# clone or system boost). Header-only -> just an include path, no link/build.
option(HYLE_BENCH_BOOST "build IntrusiveList bench vs boost::intrusive::list" OFF)
if(HYLE_BENCH_BOOST)
  set(HYLE_BOOST_DIR "${PROJECT_SOURCE_DIR}/../boost" CACHE PATH "boost tree root")
  # Accept BOTH boost layouts: monolithic (boost/intrusive/list.hpp at root, e.g.
  # system boost) OR modular superproject (libs/<lib>/include/boost/...; a fresh
  # boostorg clone with submodules checked out). boost::intrusive is header-only but
  # pulls a small dep closure (config/assert/core/move/static_assert) -- add each
  # lib's include/ dir in the modular case.
  if(EXISTS "${HYLE_BOOST_DIR}/boost/intrusive/list.hpp")
    target_include_directories(hyle_container_misc_bench SYSTEM PRIVATE "${HYLE_BOOST_DIR}")
  elseif(EXISTS "${HYLE_BOOST_DIR}/libs/intrusive/include/boost/intrusive/list.hpp")
    foreach(_bl intrusive config assert core move static_assert)
      target_include_directories(hyle_container_misc_bench SYSTEM PRIVATE
        "${HYLE_BOOST_DIR}/libs/${_bl}/include")
    endforeach()
  else()
    message(FATAL_ERROR "HYLE_BENCH_BOOST=ON but no boost/intrusive/list.hpp at ${HYLE_BOOST_DIR} (checked monolithic + modular libs/ layout; run `git submodule update --init` in the boost tree)")
  endif()
  target_compile_definitions(hyle_container_misc_bench PRIVATE HYLE_BENCH_BOOST)
endif()

# container-06c fuzzy path match bench: proves O(N) scaling (ns/candidate flat across
# N=1k/10k/100k) + races the std subsequence floor (are the fzf bonuses ~free?). Own exe.
add_executable(hyle_fuzzy_bench container/fuzzy_bench.cc)
target_link_libraries(hyle_fuzzy_bench PRIVATE hyle_container nanobench)
hyle_apply_warnings(hyle_fuzzy_bench)

# task #23: hash cluster. HYLE hash_bytes(XXH3)/mix64 vs raw XXH3 (vendored, always)
# + std::hash. wyhash is a sibling clone -> opt-in -DHYLE_BENCH_WYHASH=ON (header-only).
add_executable(hyle_hash_bench container/hash_bench.cc)
target_link_libraries(hyle_hash_bench PRIVATE hyle_container nanobench)
target_include_directories(hyle_hash_bench SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/xxhash")
hyle_apply_warnings(hyle_hash_bench)
option(HYLE_BENCH_WYHASH "build hash bench vs wyhash" OFF)
if(HYLE_BENCH_WYHASH)
  set(HYLE_WYHASH_DIR "${PROJECT_SOURCE_DIR}/../wyhash" CACHE PATH "cloned wyhash")
  if(NOT EXISTS "${HYLE_WYHASH_DIR}/wyhash.h")
    message(FATAL_ERROR "HYLE_BENCH_WYHASH=ON but no wyhash.h at ${HYLE_WYHASH_DIR}")
  endif()
  target_compile_definitions(hyle_hash_bench PRIVATE HYLE_BENCH_WYHASH)
  target_include_directories(hyle_hash_bench SYSTEM PRIVATE "${HYLE_WYHASH_DIR}")
endif()

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
  set(TBBMALLOC_BUILD ON CACHE BOOL "" FORCE)  # need tbbmalloc for scalable_allocator (arena bench)
  set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory("${HYLE_TBB_DIR}" "${CMAKE_BINARY_DIR}/onetbb" EXCLUDE_FROM_ALL)
  target_compile_definitions(hyle_queue_sota_bench PRIVATE HYLE_BENCH_TBB)
  target_link_libraries(hyle_queue_sota_bench PRIVATE TBB::tbb)
  # arena_pool_bench: add tbb's scalable_allocator as the STRONG shared-pool rival
  # (std::pmr::synchronized_pool is mutex-serialized/weak; tbb is a real MT allocator).
  # Link BOTH tbbmalloc (scalable_malloc/free) and tbb (its transitive tbb12 dep, so the
  # linker finds tbb12.lib's dir -- tbbmalloc's INTERFACE pulls tbb12 in bare otherwise).
  target_compile_definitions(hyle_arena_pool_bench PRIVATE HYLE_BENCH_TBB)
  target_link_libraries(hyle_arena_pool_bench PRIVATE TBB::tbbmalloc TBB::tbb)
endif()

# Optional: rigtorp::mpmc::Queue -- the SAME-CONTRACT rival to BoundedQueue (strict-
# bounded single-ring MPMC, the canonical SOTA for exactly that contract). Single
# header -> just an include path. Enable with -DHYLE_BENCH_RIGTORP=ON; point
# HYLE_RIGTORP_DIR at the clone (sibling `rigtorp-MPMCQueue` by default).
option(HYLE_BENCH_RIGTORP "build queue bench vs rigtorp MPMCQueue (same-contract SOTA)" OFF)
if(HYLE_BENCH_RIGTORP)
  set(HYLE_RIGTORP_DIR "${PROJECT_SOURCE_DIR}/../rigtorp-MPMCQueue" CACHE PATH "cloned rigtorp MPMCQueue")
  if(NOT EXISTS "${HYLE_RIGTORP_DIR}/include/rigtorp/MPMCQueue.h")
    message(FATAL_ERROR "HYLE_BENCH_RIGTORP=ON but no MPMCQueue.h at ${HYLE_RIGTORP_DIR}")
  endif()
  target_compile_definitions(hyle_queue_sota_bench PRIVATE HYLE_BENCH_RIGTORP)
  target_include_directories(hyle_queue_sota_bench SYSTEM PRIVATE "${HYLE_RIGTORP_DIR}/include")
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
# Per-ISA flags via the single-source hyle_set_isa (HyleModule.cmake). scalar TU: none.
hyle_set_isa(simd/simd_kernels_sse4.cc SSE42)
hyle_set_isa(simd/simd_kernels_avx2.cc AVX2)
hyle_apply_warnings(hyle_simd_bench)

# math bench: HYLE's dispatched exp/log/tanh (vectorized borrow core) vs scalar libm +
# same-ISA SOTA rivals (gap-fill #36): xsimd (-DHYLE_BENCH_XSIMD) and Highway contrib/math
# (-DHYLE_BENCH_HIGHWAY, wired in the Highway block below). Both lower to AVX2 transcendentals
# so they match HYLE's AVX2 path. The rival libs are now VENDORED in third_party (the gates
# just toggle the bench-side comparison rows). Own exe.
add_executable(hyle_math_bench simd/math_bench.cc)
target_link_libraries(hyle_math_bench PRIVATE hyle_simd nanobench)
option(HYLE_BENCH_XSIMD "build math bench vs xsimd (same-ISA transcendental SOTA)" OFF)
if(HYLE_BENCH_XSIMD)
  target_compile_definitions(hyle_math_bench PRIVATE HYLE_BENCH_XSIMD)
  target_include_directories(hyle_math_bench SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/xsimd/include")
  hyle_set_isa(simd/math_bench.cc AVX2)  # xsimd lowers to AVX2 transcendentals
endif()
hyle_apply_warnings(hyle_math_bench)

# dtype cvt bench: fp16/bf16<->f32 batch. HYLE F16C dispatch vs scalar reference (+ Highway
# gated). The *_batch ops live in hyle_simd; the bench TU only needs special ISA flags when
# the Highway path is compiled in (PromoteTo/DemoteTo intrinsics) -- added in the Highway
# block below. Own exe (math_bench owns main()).
add_executable(hyle_dtype_bench simd/dtype_bench.cc)
target_link_libraries(hyle_dtype_bench PRIVATE hyle_simd nanobench)
hyle_apply_warnings(hyle_dtype_bench)

# Optional SOTA parity: HYLE avx2 vs Highway avx2, head to head (same ISA). Highway is now
# VENDORED in third_party; -DHYLE_BENCH_HIGHWAY just toggles the comparison rows.
option(HYLE_BENCH_HIGHWAY "build the simd-vs-Highway parity bench" OFF)
if(HYLE_BENCH_HIGHWAY)
  set(_hwy "${PROJECT_SOURCE_DIR}/third_party/highway")
  target_compile_definitions(hyle_simd_bench PRIVATE HYLE_BENCH_HIGHWAY)
  target_include_directories(hyle_simd_bench SYSTEM PRIVATE "${_hwy}")
  # Highway static-dispatch resolves its width from the FULL AVX2 baseline (SSE4=PCLMUL+AES,
  # then BMI2+FMA+F16C); the bare AVX2 tier silently downgrades it to 128-bit SSSE3 on GCC/Clang,
  # which would make this "parity" bench race HYLE-256b vs Highway-128b. hyle_set_highway_isa
  # adds the missing baseline features so Highway picks true 256-bit AVX2 (see HyleModule.cmake).
  hyle_set_highway_isa(simd/simd_kernels_highway.cc)
  # dtype bench's Highway f16 PromoteTo/DemoteTo path (same gate + full-baseline ISA on its TU).
  target_compile_definitions(hyle_dtype_bench PRIVATE HYLE_BENCH_HIGHWAY)
  target_include_directories(hyle_dtype_bench SYSTEM PRIVATE "${_hwy}")
  hyle_set_highway_isa(simd/dtype_bench.cc)
  # math bench's Highway contrib/math (Exp/Log/Tanh) path -- same gate + full-baseline ISA
  # (gap-fill #36: same-ISA transcendental SOTA). Note: this OVERRIDES the xsimd AVX2 tier set
  # below/above on the same TU, but the highway baseline is a strict superset, so xsimd (which
  # only needs -mavx2) still lowers to AVX2 transcendentals -- no regression to the xsimd rows.
  target_compile_definitions(hyle_math_bench PRIVATE HYLE_BENCH_HIGHWAY)
  target_include_directories(hyle_math_bench SYSTEM PRIVATE "${_hwy}")
  hyle_set_highway_isa(simd/math_bench.cc)
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
  # absl::InlinedVector is the small-vector SOTA -> feed it to the misc bench too.
  target_compile_definitions(hyle_container_misc_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_container_misc_bench PRIVATE absl::inlined_vector)
  # absl::Status/StatusOr are the same tagged-pointer design as HYLE Status -> the
  # error bench's SOTA rival (same gate). absl::statusor pulls in absl::status.
  target_compile_definitions(hyle_error_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_bench PRIVATE absl::status absl::statusor)
endif()
