# bench: microbenchmarks (charter: nanobench). Built only with -DHYLE_BUILD_BENCH=ON.
# nanobench is a single-header lib (vendored third_party/nanobench); compile its
# impl into one static lib that bench exes link. hyle_add_bench(<mod>) (see
# HyleModule.cmake) makes hyle_<mod>_bench linked to hyle_<mod> + nanobench.
#
# Highway/xsimd are deliberately external and pinned here: benchmark rivals must
# remain unchanged while simd-20 ports their product kernels into HYLE. Product
# targets still consume their in-repo copies until the staged extraction finishes.
include(vendor-versions)

if(WIN32)
  set(_hyle_bench_vendor_default "C:/public/project-hyle")
elseif(APPLE)
  set(_hyle_bench_vendor_default "$ENV{HOME}/vendor")
else()
  # A WSL/Linux checkout is user-local; require an explicit cache or environment
  # value rather than silently assuming a sibling clone that may drift.
  set(_hyle_bench_vendor_default "$ENV{HYLE_BENCH_VENDOR_DIR}")
endif()
set(HYLE_BENCH_VENDOR_DIR "${_hyle_bench_vendor_default}" CACHE PATH
    "Directory containing pinned external highway/ and xsimd/ benchmark checkouts")
unset(_hyle_bench_vendor_default)

if(NOT HYLE_BENCH_VENDOR_DIR)
  message(FATAL_ERROR
    "HYLE_BUILD_BENCH=ON on Linux/WSL requires -DHYLE_BENCH_VENDOR_DIR=<directory>. "
    "Run HYLE_BENCH_VENDOR_DIR=<directory> scripts/fetch-bench-vendors.sh first.")
endif()

set(_hyle_bench_highway_dir "${HYLE_BENCH_VENDOR_DIR}/highway")
set(_hyle_bench_xsimd_dir "${HYLE_BENCH_VENDOR_DIR}/xsimd")

function(hyle_require_pinned_bench_vendor name directory revision required_file)
  if(NOT EXISTS "${directory}/${required_file}")
    message(FATAL_ERROR
      "HYLE_BENCH_VENDOR_DIR='${HYLE_BENCH_VENDOR_DIR}' lacks ${name}/${required_file}. "
      "Run scripts/fetch-bench-vendors.sh (or .ps1 on Windows) to populate pinned vendors.")
  endif()

  find_package(Git REQUIRED)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${directory}" rev-parse HEAD
    RESULT_VARIABLE _hyle_vendor_git_status
    OUTPUT_VARIABLE _hyle_vendor_head
    ERROR_VARIABLE _hyle_vendor_git_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _hyle_vendor_git_status EQUAL 0)
    message(FATAL_ERROR
      "${name} benchmark vendor at '${directory}' is not a Git checkout: ${_hyle_vendor_git_error}")
  endif()
  if(NOT _hyle_vendor_head STREQUAL revision)
    message(FATAL_ERROR
      "${name} benchmark vendor is ${_hyle_vendor_head}, but simd-20 requires ${revision}. "
      "Run scripts/fetch-bench-vendors.sh to restore the pinned revision.")
  endif()

  execute_process(
    COMMAND "${GIT_EXECUTABLE}" -C "${directory}" status --porcelain
    RESULT_VARIABLE _hyle_vendor_dirty_status
    OUTPUT_VARIABLE _hyle_vendor_dirty
    ERROR_VARIABLE _hyle_vendor_dirty_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _hyle_vendor_dirty_status EQUAL 0 OR NOT _hyle_vendor_dirty STREQUAL "")
    message(FATAL_ERROR
      "${name} benchmark vendor at '${directory}' has local changes. "
      "Benchmark parity requires the clean pinned checkout; run scripts/fetch-bench-vendors.sh. "
      "${_hyle_vendor_dirty_error}")
  endif()
endfunction()

hyle_require_pinned_bench_vendor(
  "Highway" "${_hyle_bench_highway_dir}" "${HYLE_BENCH_VENDOR_HIGHWAY_REV}" "hwy/highway.h")
hyle_require_pinned_bench_vendor(
  "xsimd" "${_hyle_bench_xsimd_dir}" "${HYLE_BENCH_VENDOR_XSIMD_REV}" "include/xsimd/xsimd.hpp")
message(STATUS
  "HYLE bench vendors: Highway ${HYLE_BENCH_VENDOR_HIGHWAY_REV}, xsimd ${HYLE_BENCH_VENDOR_XSIMD_REV} (${HYLE_BENCH_VENDOR_DIR})")

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
add_executable(hyle_mem_scenarios_bench mem/scenario_bench.cc)
target_link_libraries(hyle_mem_scenarios_bench PRIVATE hyle_mem nanobench
                                                       mimalloc_static)
if(WIN32)
  target_link_libraries(hyle_mem_scenarios_bench PRIVATE psapi)
endif()
hyle_apply_warnings(hyle_mem_scenarios_bench)

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

  # Optional SOTA parity #2: echo_bench's HYLE rows vs asio + libuv -- the "vs a mature
  # async FRAMEWORK" comparison (liburing above is "vs the raw API"; these are "vs the
  # frameworks everyone actually uses"). Same 64-conn/64B ping-pong, same nanobench run
  # for a trustworthy relative column.
  #  * asio (header-only, Kohlhoff standalone clone): io_uring backend, so it LINKS
  #    liburing -- enabling HYLE_BENCH_ASIO force-enables the liburing build above.
  #  * libuv (CMake project, sibling clone): epoll backend, built as its uv_a static lib.
  option(HYLE_BENCH_ASIO "build echo bench vs asio (io_uring, async-framework SOTA)" OFF)
  if(HYLE_BENCH_ASIO AND UNIX AND NOT APPLE)
    set(HYLE_ASIO_DIR "${PROJECT_SOURCE_DIR}/../asio" CACHE PATH "cloned asio (Kohlhoff)")
    if(NOT EXISTS "${HYLE_ASIO_DIR}/asio/include/asio.hpp")
      message(FATAL_ERROR "HYLE_BENCH_ASIO=ON but no asio at ${HYLE_ASIO_DIR}/asio/include")
    endif()
    # asio's io_uring backend needs liburing linked. Reuse the liburing block's clone:
    # configure+make it here if HYLE_BENCH_LIBURING didn't already.
    set(HYLE_LIBURING_DIR "${PROJECT_SOURCE_DIR}/../liburing" CACHE PATH "cloned liburing")
    if(NOT EXISTS "${HYLE_LIBURING_DIR}/src/liburing.a")
      execute_process(COMMAND ./configure WORKING_DIRECTORY "${HYLE_LIBURING_DIR}"
        RESULT_VARIABLE _lu_cfg OUTPUT_QUIET ERROR_QUIET)
      execute_process(COMMAND make -j -C src WORKING_DIRECTORY "${HYLE_LIBURING_DIR}"
        RESULT_VARIABLE _lu_make OUTPUT_QUIET ERROR_QUIET)
      if(NOT EXISTS "${HYLE_LIBURING_DIR}/src/liburing.a")
        message(FATAL_ERROR "asio needs liburing; build failed in ${HYLE_LIBURING_DIR}")
      endif()
    endif()
    target_compile_definitions(hyle_io_bench PRIVATE HYLE_BENCH_ASIO
      ASIO_HAS_IO_URING ASIO_DISABLE_EPOLL ASIO_STANDALONE)
    target_include_directories(hyle_io_bench SYSTEM PRIVATE
      "${HYLE_ASIO_DIR}/asio/include" "${HYLE_LIBURING_DIR}/src/include")
    target_link_libraries(hyle_io_bench PRIVATE "${HYLE_LIBURING_DIR}/src/liburing.a")
  endif()

  option(HYLE_BENCH_LIBUV "build echo bench vs libuv (epoll, async-runtime SOTA)" OFF)
  if(HYLE_BENCH_LIBUV AND UNIX AND NOT APPLE)
    set(HYLE_LIBUV_DIR "${PROJECT_SOURCE_DIR}/../libuv" CACHE PATH "cloned libuv")
    if(NOT EXISTS "${HYLE_LIBUV_DIR}/include/uv.h")
      message(FATAL_ERROR "HYLE_BENCH_LIBUV=ON but no libuv at ${HYLE_LIBUV_DIR}")
    endif()
    # libuv is a CMake project; add its static target (uv_a) EXCLUDE_FROM_ALL.
    if(NOT TARGET uv_a)
      set(LIBUV_BUILD_TESTS OFF CACHE BOOL "" FORCE)
      set(LIBUV_BUILD_BENCH OFF CACHE BOOL "" FORCE)
      add_subdirectory("${HYLE_LIBUV_DIR}" "${CMAKE_BINARY_DIR}/libuv" EXCLUDE_FROM_ALL)
    endif()
    target_compile_definitions(hyle_io_bench PRIVATE HYLE_BENCH_LIBUV)
    target_include_directories(hyle_io_bench SYSTEM PRIVATE "${HYLE_LIBUV_DIR}/include")
    target_link_libraries(hyle_io_bench PRIVATE uv_a)
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

  # io connection-lifecycle bench: the REALISTIC full-connection cost (accept + serve R
  # round-trips + close) that echo_bench (pre-built socketpairs, never accept/close) and
  # accept_bench (accept-then-drop, no service) each factor out. Sweeps rounds/conn =
  # 1/4/16 to walk the ephemeral (setup-dominated) -> persistent (steady-state) spectrum.
  # Real loopback TCP (real handshake + close), so hyle_io (tcp wrappers) + ws2_32 on Win.
  add_executable(hyle_conn_lifecycle_bench io/conn_lifecycle_bench.cc)
  target_link_libraries(hyle_conn_lifecycle_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_conn_lifecycle_bench)
  if(WIN32)
    target_link_libraries(hyle_conn_lifecycle_bench PRIVATE ws2_32)
  endif()

  # io HTTP bench (io-10): request/response throughput of the FULL HTTP/1.1 path
  # (picohttpparser parse + GrowBuf head/body copies + Request/Response build + status
  # /header formatting) vs a RAW TCP echo over the SAME co_await TcpStream transport
  # moving the IDENTICAL byte volume with no HTTP work. The ratio isolates the HTTP
  # parse/framing layer cost from the io transport (already measured by echo_bench),
  # and the 1/2/4/8-worker sweep shows the HTTP path preserves shard-per-core scaling.
  # Socketpair transport (like echo_bench) -- no accept, no TIME_WAIT. Needs hyle_io
  # (http/tcp) + ws2_32 on Windows.
  add_executable(hyle_http_bench io/http_bench.cc)
  target_link_libraries(hyle_http_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_http_bench)
  if(WIN32)
    target_link_libraries(hyle_http_bench PRIVATE ws2_32)
  endif()

  # io multishot-recv bench (io-07b): does the multishot + provided-buffer-ring
  # awaitable beat one-shot RecvOp on a receive-heavy load? Same-worker sender+receiver
  # fibers + DGRAM socketpair (no cross-thread wakeup floor, equal completion counts);
  # the only variable is the receiver's recv method. Isolates the per-recv-submit cost
  # multishot removes. Linux/io_uring only in effect (register_buf_ring returns false
  # elsewhere -> multishot row self-skips); compiles everywhere. Same sched dep.
  add_executable(hyle_multishot_bench io/multishot_bench.cc)
  target_link_libraries(hyle_multishot_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_multishot_bench)
  if(WIN32)
    target_link_libraries(hyle_multishot_bench PRIVATE ws2_32)
  endif()

  # io multishot ZERO-COPY ECHO bench (io-07c承重): the workload multishot is FOR --
  # echo each received chunk straight back FROM the ring buffer (no copy) vs a one-shot
  # recv-into-buffer + send. Where the recv-drain bench tied, this should show
  # multishot's zero-copy + no-per-recv-submit win. Same sched dep; Linux/io_uring only.
  add_executable(hyle_multishot_echo_bench io/multishot_echo_bench.cc)
  target_link_libraries(hyle_multishot_echo_bench PRIVATE hyle_sched hyle_io nanobench)
  hyle_apply_warnings(hyle_multishot_echo_bench)
  if(WIN32)
    target_link_libraries(hyle_multishot_echo_bench PRIVATE ws2_32)
  endif()

  # io TLS bench (io-08): TLS handshake/s + encrypted-echo throughput over HYLE's
  # memory-BIO pump vs RAW OpenSSL over a blocking socket (same cert/cipher/payload).
  # The ratio isolates HYLE's co_await pump + fiber overhead from the identical
  # OpenSSL crypto floor both pay. Only when OpenSSL is present (same OpenSSL_FOUND
  # gate as the tls.cc build). Links OpenSSL + ws2_32 on Windows.
  if(OpenSSL_FOUND)
    add_executable(hyle_io_tls_bench io/tls_bench.cc)
    target_link_libraries(hyle_io_tls_bench PRIVATE hyle_sched hyle_io nanobench
                          OpenSSL::SSL OpenSSL::Crypto)
    hyle_apply_warnings(hyle_io_tls_bench)
    if(WIN32)
      target_link_libraries(hyle_io_tls_bench PRIVATE ws2_32)
    endif()
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

# time-02 timer wheel bench: the wheel vs an indexed eager-erase min-heap (the
# actual libevent/libuv shape), plus CommonTimeoutQueue vs an embedded intrusive
# TAILQ (libevent common-timeout shape). Own exe.
add_executable(hyle_timer_wheel_bench time/timer_wheel_bench.cc)
target_link_libraries(hyle_timer_wheel_bench PRIVATE hyle_time nanobench)
hyle_apply_warnings(hyle_timer_wheel_bench)

# sched-05 fiber-aware sync primitives bench (own exe: scheduler_bench.cc already
# owns main(), so this is a second target rather than another source in it).
add_executable(hyle_sched_sync_bench sched/sync_bench.cc)
target_link_libraries(hyle_sched_sync_bench PRIVATE hyle_sched nanobench)
hyle_apply_warnings(hyle_sched_sync_bench)

# sched-09 concurrency-primitive bench: Semaphore/RwLock/oneshot/watch/broadcast
# (the sched-09 additions sync_bench doesn't cover). Own exe. Uses Scheduler+Fiber
# so it needs fcontext like the rest of the fiber-driven sched benches.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_sync_ext_bench sched/sync_ext_bench.cc)
  target_link_libraries(hyle_sched_sync_ext_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_sync_ext_bench)
endif()

# sched-11 spawn_blocking bench: the blocking-offload pool round-trip + fan-out. HYLE
# rows always build (into hyle_sched); the marl blocking_call comparison row is gated
# -DHYLE_BENCH_MARL (wired in the marl block below). Own exe; needs fcontext.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_blocking_bench sched/blocking_bench.cc)
  target_link_libraries(hyle_sched_blocking_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_blocking_bench)
endif()

# sched-07 cancellation-tree bench: is_cancelled() hot-path (vs bare atomic) +
# cancel() propagation O(N) (wide + deep trees) + derive() per-child cost. Own exe;
# CancelScope references Fiber so it needs fcontext like the rest of sched.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_cancel_bench sched/cancel_bench.cc)
  target_link_libraries(hyle_sched_cancel_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_cancel_bench)
endif()

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
# API). Heavy external build -> add_subdirectory the local oneTBB source snapshot.
# Enable with -DHYLE_BENCH_TBB=ON; point HYLE_TBB_DIR at that snapshot (sibling
# by default).
option(HYLE_BENCH_TBB "build queue/vector bench vs oneTBB" OFF)
if(HYLE_BENCH_TBB)
  set(HYLE_TBB_DIR "${PROJECT_SOURCE_DIR}/../oneTBB" CACHE PATH
      "local oneTBB source snapshot")
  if(NOT EXISTS "${HYLE_TBB_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_TBB=ON but no oneTBB at ${HYLE_TBB_DIR}")
  endif()
  set(TBB_TEST OFF CACHE BOOL "" FORCE)
  set(TBBMALLOC_BUILD ON CACHE BOOL "" FORCE)  # need tbbmalloc for scalable_allocator (arena bench)
  set(TBB_EXAMPLES OFF CACHE BOOL "" FORCE)
  add_subdirectory("${HYLE_TBB_DIR}" "${CMAKE_BINARY_DIR}/onetbb" EXCLUDE_FROM_ALL)
  target_compile_definitions(hyle_queue_sota_bench PRIVATE HYLE_BENCH_TBB)
  target_link_libraries(hyle_queue_sota_bench PRIVATE TBB::tbb)
  # mem_scenarios_bench: oneTBB fixed_pool is the direct fixed-buffer rival;
  # scalable_allocator remains a growable throughput reference. memory_pool.h is a
  # oneTBB preview header and requires TBB_PREVIEW_MEMORY_POOL=1.
  target_compile_definitions(hyle_mem_scenarios_bench PRIVATE
                             HYLE_BENCH_TBB TBB_PREVIEW_MEMORY_POOL=1)
  target_link_libraries(hyle_mem_scenarios_bench PRIVATE TBB::tbbmalloc TBB::tbb)
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
# so they match HYLE's AVX2 path. The rivals come from the separately pinned external
# checkouts; the gates only toggle their bench-side comparison rows. Own exe.
add_executable(hyle_math_bench simd/math_bench.cc)
target_link_libraries(hyle_math_bench PRIVATE hyle_simd nanobench)
option(HYLE_BENCH_XSIMD "build math bench vs xsimd (same-ISA transcendental SOTA)" OFF)
option(HYLE_BENCH_HIGHWAY "build math bench vs Highway (same-ISA transcendental SOTA)" OFF)
if(HYLE_BENCH_XSIMD)
  target_compile_definitions(hyle_math_bench PRIVATE HYLE_BENCH_XSIMD)
  target_include_directories(hyle_math_bench SYSTEM PRIVATE "${_hyle_bench_xsimd_dir}/include")
endif()
if(HYLE_BENCH_HIGHWAY)
  target_compile_definitions(hyle_math_bench PRIVATE HYLE_BENCH_HIGHWAY)
  target_include_directories(hyle_math_bench SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
endif()
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  if(HYLE_BENCH_XSIMD OR HYLE_BENCH_HIGHWAY)
    hyle_set_highway_isa(simd/math_bench.cc)
  endif()
else()
  # Clear an x86-only source property left by a prior configuration of this build directory.
  hyle_clear_source_isa("${CMAKE_CURRENT_SOURCE_DIR}/simd/math_bench.cc")
endif()
hyle_apply_warnings(hyle_math_bench)

# simd-12 scaling benches: every rival is entered through an explicit noinline wrapper in the same
# fixed-ISA TU. This separates public/direct dispatch tax from vendor/candidate algorithm cost.
add_executable(hyle_math_scaling_bench simd/math_scaling_bench.cc)
target_link_libraries(hyle_math_scaling_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_math_scaling_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_math_scaling_bench SYSTEM PRIVATE
  "${_hyle_bench_xsimd_dir}/include")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_isa(simd/math_scaling_bench.cc AVX2)
endif()
hyle_apply_warnings(hyle_math_scaling_bench)

add_executable(hyle_bitpack_bench simd/bitpack_bench.cc)
target_link_libraries(hyle_bitpack_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_bitpack_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_bitpack_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
# The direct vendor comparison remains AVX2-fixed. Compile the HYLE u16 correction benchmark at
# the same ISA and select its matching D-tag backend, avoiding an inline/vector-width asymmetry.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_isa(simd/bitpack_bench.cc)
  set_source_files_properties(simd/bitpack_bench.cc PROPERTIES
    COMPILE_DEFINITIONS "HYLE_SIMD_TARGET_AVX2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_isa(simd/bitpack_bench.cc NEON)
  set_source_files_properties(simd/bitpack_bench.cc PROPERTIES
    COMPILE_DEFINITIONS "HYLE_SIMD_TARGET_NEON")
endif()
hyle_apply_warnings(hyle_bitpack_bench)

# AI activation + softmax + RNG throughput (directive3 P1 blind-spot fill). The activation
# *_f32_batch come from hyle_simd (runtime-dispatched); the inline VectorXoshiro takes THIS TU's
# vec.h backend, so pin HYLE_SIMD_TARGET_AVX2 on x86 to measure the real 256-bit RNG (else the
# default 128-bit backend makes VectorXoshiro look slower than std::mt19937).
add_executable(hyle_activation_bench simd/activation_bench.cc)
target_link_libraries(hyle_activation_bench PRIVATE hyle_simd nanobench)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_compile_definitions(hyle_activation_bench PRIVATE HYLE_SIMD_TARGET_AVX2)
  hyle_set_isa(simd/activation_bench.cc AVX2)
endif()
hyle_apply_warnings(hyle_activation_bench)

# P1-1 prefix scan: scalar O(n) dependence chain vs HYLE Kogge-Stone inclusive_scan.
add_executable(hyle_scan_bench simd/scan_bench.cc)
target_link_libraries(hyle_scan_bench PRIVATE hyle_simd nanobench)
hyle_apply_warnings(hyle_scan_bench)

add_executable(hyle_search_bench simd/search_bench.cc)
target_link_libraries(hyle_search_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_search_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_isa(simd/search_bench.cc)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_isa(simd/search_bench.cc NEON)
endif()
hyle_apply_warnings(hyle_search_bench)

add_executable(hyle_mixed_dot_bench simd/mixed_dot_bench.cc)
target_link_libraries(hyle_mixed_dot_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_mixed_dot_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_isa(simd/mixed_dot_bench.cc)
  set_source_files_properties(simd/mixed_dot_bench.cc PROPERTIES
    COMPILE_DEFINITIONS "HYLE_SIMD_TARGET_AVX2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  hyle_set_isa(simd/mixed_dot_bench.cc NEON)
  set_source_files_properties(simd/mixed_dot_bench.cc PROPERTIES
    COMPILE_DEFINITIONS "HYLE_SIMD_TARGET_NEON")
endif()
hyle_apply_warnings(hyle_mixed_dot_bench)

add_executable(hyle_sort_bench simd/sort_bench.cc)
target_link_libraries(hyle_sort_bench PRIVATE hyle_simd nanobench)
target_include_directories(hyle_sort_bench PRIVATE "${PROJECT_SOURCE_DIR}")
target_include_directories(hyle_sort_bench SYSTEM PRIVATE
  "${_hyle_bench_highway_dir}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_set_highway_isa(simd/sort_bench.cc)
endif()
hyle_apply_warnings(hyle_sort_bench)

# P1-2: radix vs std::sort / HYLE sort auto-threshold crossover.
add_executable(hyle_radix_sort_bench simd/radix_sort_bench.cc)
target_link_libraries(hyle_radix_sort_bench PRIVATE hyle_simd nanobench)
hyle_apply_warnings(hyle_radix_sort_bench)

# gather/compress native-vs-emulated bench (simd-09 #66/#69, "measure don't assume"). Header-only
# (includes vec.h at a fixed ISA target), so per-ISA exes like the vec_test variants: avx2 runs on
# an AVX2 host, avx512 under Intel SDE. Only x86 (native gather/compress are x86 intrinsics).
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  hyle_isa_flags(_gb_avx2 AVX2)
  add_executable(hyle_gather_bench_avx2 simd/gather_bench.cc)
  target_link_libraries(hyle_gather_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_gather_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(hyle_gather_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  target_compile_options(hyle_gather_bench_avx2 PRIVATE ${_gb_avx2})
  hyle_apply_warnings(hyle_gather_bench_avx2)

  hyle_isa_flags(_gb_avx512 AVX512)
  add_executable(hyle_gather_bench_avx512 simd/gather_bench.cc)
  target_link_libraries(hyle_gather_bench_avx512 PRIVATE nanobench)
  target_include_directories(hyle_gather_bench_avx512 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_compile_definitions(hyle_gather_bench_avx512 PRIVATE HYLE_SIMD_TARGET_AVX512)
  target_compile_options(hyle_gather_bench_avx512 PRIVATE ${_gb_avx512})
  hyle_apply_warnings(hyle_gather_bench_avx512)

  # op-parity bench (HYLE self-made ops vs Highway same op, same ISA, same TU -> apples-to-apples,
  # both inline). Answers "did self-making an op cost us vs just calling Highway?". Per-ISA exe:
  # avx2 on host, avx512 under SDE. Needs the Highway include dir AND the FULL AVX2/AVX512 baseline
  # (hyle_set_highway_isa / hyle_set_avx512_*_isa) so Highway's static dispatch doesn't silently
  # downgrade to SSSE3 (which would make it a HYLE-256 vs Highway-128 non-race).
  add_executable(hyle_op_parity_bench_avx2 simd/op_parity_bench.cc)
  target_link_libraries(hyle_op_parity_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}" "${_hyle_bench_xsimd_dir}/include")
  target_compile_definitions(hyle_op_parity_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  # PER-TARGET flags (not set_source_files_properties, which would leak onto the avx512 target that
  # shares this source): full AVX2 baseline so Highway static dispatch picks true 256-bit.
  if(MSVC)
    target_compile_options(hyle_op_parity_bench_avx2 PRIVATE /arch:AVX2 /bigobj)
  else()
    target_compile_options(hyle_op_parity_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_op_parity_bench_avx2)

  add_executable(hyle_op_parity_bench_avx512 simd/op_parity_bench.cc)
  target_link_libraries(hyle_op_parity_bench_avx512 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_bench_avx512 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_bench_avx512 SYSTEM PRIVATE "${_hyle_bench_highway_dir}" "${_hyle_bench_xsimd_dir}/include")
  target_compile_definitions(hyle_op_parity_bench_avx512 PRIVATE HYLE_SIMD_TARGET_AVX512)
  hyle_isa_flags(_op_avx512 AVX512)
  target_compile_options(hyle_op_parity_bench_avx512 PRIVATE ${_op_avx512})
  hyle_apply_warnings(hyle_op_parity_bench_avx512)

  # sse4 variant (128-bit): proves the sse4 backend's native-op rewrites (PopCount/LZCnt/TZCnt/
  # ReverseBits/DupEven/Odd/ConcatEven/Odd/GetExponent) match/beat Highway's SSE4 same ops. Same
  # apples-to-apples (both inline, same TU/ISA). Highway needs the AES/PCLMUL baseline so its SSE4
  # static target isn't downgraded.
  add_executable(hyle_op_parity_bench_sse4 simd/op_parity_bench.cc)
  target_link_libraries(hyle_op_parity_bench_sse4 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_bench_sse4 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_bench_sse4 SYSTEM PRIVATE "${_hyle_bench_highway_dir}" "${_hyle_bench_xsimd_dir}/include")
  target_compile_definitions(hyle_op_parity_bench_sse4 PRIVATE HYLE_SIMD_TARGET_SSE4)
  if(MSVC)
    # MSVC has no /arch:SSE4; SSE4.2 intrinsics are unconditional on x64.
  else()
    target_compile_options(hyle_op_parity_bench_sse4 PRIVATE -msse4.2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_op_parity_bench_sse4)

  add_executable(hyle_op_parity_full_bench_avx2 simd/op_parity_full_bench.cc)
  target_link_libraries(hyle_op_parity_full_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_full_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_full_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_op_parity_full_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  if(MSVC)
    target_compile_options(hyle_op_parity_full_bench_avx2 PRIVATE /arch:AVX2 /bigobj)
  else()
    target_compile_options(hyle_op_parity_full_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_op_parity_full_bench_avx2)

  # widen / promote-demote / MAC family (bespoke harness; was un-wired)
  add_executable(hyle_op_parity_widen_bench_avx2 simd/op_parity_widen_bench.cc)
  target_link_libraries(hyle_op_parity_widen_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_op_parity_widen_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_widen_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_op_parity_widen_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  if(MSVC)
    target_compile_options(hyle_op_parity_widen_bench_avx2 PRIVATE /arch:AVX2 /bigobj)
  else()
    target_compile_options(hyle_op_parity_widen_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_op_parity_widen_bench_avx2)

  # Stream is a non-temporal-store algorithm contract, measured on a >LLC working set rather
  # than in the L1 op-parity harness.
  add_executable(hyle_stream_bench_avx2 simd/stream_bench.cc)
  target_include_directories(hyle_stream_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_stream_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_stream_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  if(MSVC)
    target_compile_options(hyle_stream_bench_avx2 PRIVATE /arch:AVX2)
  else()
    target_compile_options(hyle_stream_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_stream_bench_avx2)

  # crypto AES/CLMul (Tier1; needs AES-NI + PCLMUL baseline)
  add_executable(hyle_crypto_bench_avx2 simd/crypto_bench.cc)
  target_link_libraries(hyle_crypto_bench_avx2 PRIVATE nanobench)
  target_include_directories(hyle_crypto_bench_avx2 PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_crypto_bench_avx2 SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_crypto_bench_avx2 PRIVATE HYLE_SIMD_TARGET_AVX2)
  if(MSVC)
    target_compile_options(hyle_crypto_bench_avx2 PRIVATE /arch:AVX2)
  else()
    target_compile_options(hyle_crypto_bench_avx2 PRIVATE -mavx2 -mfma -mf16c -mbmi -mbmi2 -maes -mpclmul)
  endif()
  hyle_apply_warnings(hyle_crypto_bench_avx2)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
  add_executable(hyle_op_parity_full_bench_neon simd/op_parity_full_bench.cc)
  target_link_libraries(hyle_op_parity_full_bench_neon PRIVATE nanobench)
  target_include_directories(hyle_op_parity_full_bench_neon PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_full_bench_neon SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_op_parity_full_bench_neon PRIVATE HYLE_SIMD_TARGET_NEON)
  hyle_apply_warnings(hyle_op_parity_full_bench_neon)

  # NEON parity for the mask/gather/tail/crypto TU (op_parity_bench) and the widen TU
  # (bf16/U8/promote-demote). These were x86-only -- the recently-added masked-FMA / bit-ternary /
  # masked-gather / bf16-dot / U8FromU32 / narrow-reduce ops had NO NEON perf bench (only gtest).
  # gather/crypto Highway paths reference hwy::Abort (HWY_DASSERT) that Apple-clang doesn't DCE on
  # arm64 -> compile highway's abort.cc in (op_parity_full doesn't hit it, so it links bare).
  add_executable(hyle_op_parity_bench_neon simd/op_parity_bench.cc
                 "${_hyle_bench_highway_dir}/hwy/abort.cc")
  target_link_libraries(hyle_op_parity_bench_neon PRIVATE nanobench)
  target_include_directories(hyle_op_parity_bench_neon PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_bench_neon SYSTEM PRIVATE "${_hyle_bench_highway_dir}" "${_hyle_bench_xsimd_dir}/include")
  target_compile_definitions(hyle_op_parity_bench_neon PRIVATE HYLE_SIMD_TARGET_NEON)
  hyle_apply_warnings(hyle_op_parity_bench_neon)

  add_executable(hyle_op_parity_widen_bench_neon simd/op_parity_widen_bench.cc
                 "${_hyle_bench_highway_dir}/hwy/abort.cc")
  target_link_libraries(hyle_op_parity_widen_bench_neon PRIVATE nanobench)
  target_include_directories(hyle_op_parity_widen_bench_neon PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_op_parity_widen_bench_neon SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_op_parity_widen_bench_neon PRIVATE HYLE_SIMD_TARGET_NEON)
  hyle_apply_warnings(hyle_op_parity_widen_bench_neon)

  add_executable(hyle_stream_bench_neon simd/stream_bench.cc)
  target_include_directories(hyle_stream_bench_neon PRIVATE "${PROJECT_SOURCE_DIR}/include")
  target_include_directories(hyle_stream_bench_neon SYSTEM PRIVATE "${_hyle_bench_highway_dir}")
  target_compile_definitions(hyle_stream_bench_neon PRIVATE HYLE_SIMD_TARGET_NEON)
  hyle_apply_warnings(hyle_stream_bench_neon)
endif()

# dtype cvt bench: fp16/bf16<->f32 batch. HYLE F16C dispatch vs scalar reference (+ Highway
# gated). The *_batch ops live in hyle_simd; the bench TU only needs special ISA flags when
# the Highway path is compiled in (PromoteTo/DemoteTo intrinsics) -- added in the Highway
# block below. Own exe (math_bench owns main()).
add_executable(hyle_dtype_bench simd/dtype_bench.cc)
target_link_libraries(hyle_dtype_bench PRIVATE hyle_simd nanobench)
hyle_apply_warnings(hyle_dtype_bench)

# Optional SOTA parity: HYLE avx2 vs the separately pinned Highway checkout, head to head
# at the same ISA. -DHYLE_BENCH_HIGHWAY only toggles the comparison rows.
option(HYLE_BENCH_HIGHWAY "build the simd-vs-Highway parity bench" OFF)
if(HYLE_BENCH_HIGHWAY)
  set(_hwy "${_hyle_bench_highway_dir}")
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
  # math bench's Highway contrib/math comparison is configured beside its target above. Do not
  # force its x86-only ISA flags here: HYLE_BENCH_HIGHWAY is also valid on NEON.
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
  # sched-11 blocking_bench's marl comparison row (marl::blocking_call = new-thread-
  # per-call; the HYLE-pool-vs-marl-new-thread head-to-head). Same gate; links marl.
  if(TARGET hyle_sched_blocking_bench)
    target_compile_definitions(hyle_sched_blocking_bench PRIVATE HYLE_BENCH_MARL)
    target_link_libraries(hyle_sched_blocking_bench PRIVATE marl)
  endif()
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
  # StringBuf's strongest same-contract peer is a sharded, locked
  # absl::flat_hash_set interner (not an unlocked std map floor).
  target_compile_definitions(hyle_container_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_container_bench PRIVATE absl::flat_hash_set)
  # absl::Status/StatusOr are the same tagged-pointer design as HYLE Status -> the
  # error bench's SOTA rival (same gate). absl::statusor pulls in absl::status.
  target_compile_definitions(hyle_error_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_bench PRIVATE absl::status absl::statusor)
endif()

# compute bench (compute-06 #16): CPU submit cost, launch (re-record) vs replay (reuse a
# recorded command buffer). Vulkan-only (record/replay is a Vulkan optimization); gate on it.
# Reuses the SAME embedded saxpy SPIR-V the test uses, regenerated here via hyle_embed_shader.
if(HYLE_COMPUTE_VULKAN_ENABLED)
  hyle_embed_shader(saxpy.comp saxpy.spv.h kSaxpySpirv hyle_test_shaders)
  add_executable(hyle_compute_bench compute/compute_bench.cc "${HYLE_SHADER_kSaxpySpirv}")
  target_link_libraries(hyle_compute_bench PRIVATE hyle_compute nanobench)
  target_include_directories(hyle_compute_bench PRIVATE "${HYLE_SHADER_DIR}")
  hyle_apply_warnings(hyle_compute_bench)
endif()
