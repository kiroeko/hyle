# sched bench: fiber switch (~23ns) + spawn throughput vs std::async (~7-9x). Now
# compiled into the default bench build (no longer withheld). It once crashed at
# teardown with mimalloc ON, but an exhaustive hunt (docs/debug/ki-1-crash-hunt.md)
# RULED OUT a HYLE logic bug -- TSan (no source race) + MI_DEBUG=3 full heap checks
# (no corruption) are clean under churn + oversubscription + 24G memory pressure,
# and the original crasher is NOT reproducible on native ext4. Evidence (SIGILL +
# clean instrumented runs) points to a 9p/WSL paging artifact, not this code. Kept
# IN the build on purpose: if it ever recurs, we investigate head-on, not hide it.
  hyle_add_bench(sched SOURCES scheduler_bench.cc)

# Dependency-upgrade evidence: compare the consumer-facing Fiber resume/yield
# path with a raw round trip through the exact same vendored Boost.Context 1.92
# fcontext assembly.  The source intentionally includes sched's private shim;
# this target is validation-only and does not make fcontext public API.
if(HYLE_HAS_FCONTEXT)
  # Reuse the exact assembly source list already selected for hyle_sched instead
  # of maintaining a second architecture table in the benchmark layer.
  get_target_property(_hyle_sched_sources hyle_sched SOURCES)
  set(_hyle_bench_fcontext_asm)
  foreach(_hyle_sched_source IN LISTS _hyle_sched_sources)
    if(_hyle_sched_source MATCHES "\\.(asm|S)$")
      list(APPEND _hyle_bench_fcontext_asm "${_hyle_sched_source}")
    endif()
  endforeach()
  if(NOT _hyle_bench_fcontext_asm)
    message(FATAL_ERROR
      "hyle_sched_fcontext_bench could not recover hyle_sched's selected fcontext assembly")
  endif()
  if(HYLE_STATIC)
    # The final executable already receives the selected assembly from the
    # static hyle_sched archive.  Reuse those symbols so both HYLE and the raw
    # seam have ordinary in-image calls and no duplicate assembly definition.
    add_library(hyle_bench_fcontext_peer STATIC fcontext_peer.cc)
    target_compile_definitions(hyle_bench_fcontext_peer PUBLIC
      HYLE_BENCH_PEER_STATIC)
  else()
    # Keep raw symbols in a separate image from shared hyle_sched.  This avoids
    # relying on private product exports while preserving the exact asm bytes.
    add_library(hyle_bench_fcontext_peer SHARED
      fcontext_peer.cc ${_hyle_bench_fcontext_asm})
    target_compile_definitions(hyle_bench_fcontext_peer PRIVATE
      HYLE_BENCH_PEER_EXPORTS)
  endif()
  target_include_directories(hyle_bench_fcontext_peer PRIVATE
    "${PROJECT_SOURCE_DIR}/src/sched")
  target_link_libraries(hyle_bench_fcontext_peer PRIVATE hyle_mem)
  if(MSVC)
    target_compile_definitions(hyle_bench_fcontext_peer PRIVATE
      BOOST_CONTEXT_EXPORT=EXPORT)
  endif()
  hyle_apply_warnings(hyle_bench_fcontext_peer)

  add_executable(hyle_sched_fcontext_bench fcontext_bench.cc)
  target_link_libraries(hyle_sched_fcontext_bench PRIVATE
    hyle_sched hyle_bench_fcontext_peer nanobench)
  hyle_apply_warnings(hyle_sched_fcontext_bench)
  unset(_hyle_sched_sources)
  unset(_hyle_bench_fcontext_asm)
  unset(_hyle_sched_source)
endif()

# sched-05 fiber-aware sync primitives bench (own exe: scheduler_bench.cc already
# owns main(), so this is a second target rather than another source in it).
add_executable(hyle_sched_sync_bench sync_bench.cc)
target_link_libraries(hyle_sched_sync_bench PRIVATE hyle_sched nanobench)
hyle_apply_warnings(hyle_sched_sync_bench)

# sched-09 concurrency-primitive bench: Semaphore/RwLock/oneshot/watch/broadcast
# (the sched-09 additions sync_bench doesn't cover). Own exe. Uses Scheduler+Fiber
# so it needs fcontext like the rest of the fiber-driven sched benches.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_sync_ext_bench sync_ext_bench.cc)
  target_link_libraries(hyle_sched_sync_ext_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_sync_ext_bench)
endif()

# sched-11 spawn_blocking bench: the blocking-offload pool round-trip + fan-out. HYLE
# rows always build (into hyle_sched); the marl blocking_call comparison row is gated
# -DHYLE_BENCH_MARL (wired in the marl block below). Own exe; needs fcontext.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_blocking_bench blocking_bench.cc)
  target_link_libraries(hyle_sched_blocking_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_blocking_bench)
endif()

# sched-07 cancellation-tree bench: is_cancelled() hot-path (vs bare atomic) +
# cancel() propagation O(N) (wide + deep trees) + derive() per-child cost. Own exe;
# CancelScope references Fiber so it needs fcontext like the rest of sched.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_cancel_bench cancel_bench.cc)
  target_link_libraries(hyle_sched_cancel_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_cancel_bench)
endif()

# sched-10 JoinHandle/spawn_task/when_all/when_any bench: proves the structured-
# concurrency layer is ~zero-overhead vs the raw fiber+WaitGroup(1)+slot it wraps.
# Own exe (scheduler_bench owns main). Needs fcontext like the rest of sched.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_join_bench join_bench.cc)
  target_link_libraries(hyle_sched_join_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_join_bench)
endif()

# scheduler-layer gap benches (work-stealing scaling, spawn_on affinity,
# coroutine glue vs raw fiber spine, fiber sleep). Complements scheduler_bench
# (raw primitives) + queue_sota_bench (the work-stealing DATA STRUCTURE) by
# measuring SCHEDULER-LEVEL behaviour. Own exe. Needs fcontext (coro + timer).
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_sched_ext_bench scheduler_ext_bench.cc)
  target_link_libraries(hyle_sched_ext_bench PRIVATE hyle_sched nanobench)
  hyle_apply_warnings(hyle_sched_ext_bench)
endif()

# -DHYLE_BENCH_MARL=ON; point HYLE_MARL_DIR at a cloned marl (sibling default).
option(HYLE_BENCH_MARL "build the fiber-sync-vs-marl parity bench" OFF)
if(HYLE_BENCH_MARL)
  set(HYLE_MARL_DIR "${PROJECT_SOURCE_DIR}/../marl" CACHE PATH "cloned marl")
  if(NOT EXISTS "${HYLE_MARL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_MARL=ON but no marl at ${HYLE_MARL_DIR}")
  endif()
  # Marl declares the generic ASM language unconditionally, even though its
  # MSVC source list contains only C++. CMake 4.4's CMP0194 correctly stops
  # treating cl.exe as a generic assembler, so adding Marl's project directly
  # cannot generate with a modern MSVC + Ninja toolchain. Keep this
  # benchmark-only adapter local: build the exact MSVC source list without
  # rewriting the external checkout, while other toolchains continue to use
  # Marl's own project and architecture-specific assembly selection.
  if(MSVC)
    add_library(hyle_bench_marl STATIC
      "${HYLE_MARL_DIR}/src/debug.cpp"
      "${HYLE_MARL_DIR}/src/memory.cpp"
      "${HYLE_MARL_DIR}/src/scheduler.cpp"
      "${HYLE_MARL_DIR}/src/thread.cpp"
      "${HYLE_MARL_DIR}/src/trace.cpp")
    target_include_directories(hyle_bench_marl SYSTEM PUBLIC
      "${HYLE_MARL_DIR}/include")
    target_compile_definitions(hyle_bench_marl PRIVATE
      _CRT_SECURE_NO_WARNINGS
      NOMINMAX
      WIN32_LEAN_AND_MEAN)
    target_compile_options(hyle_bench_marl PRIVATE /wd4127 /wd4324)
    target_link_libraries(hyle_bench_marl PUBLIC Threads::Threads)
    set(_hyle_bench_marl_target hyle_bench_marl)
  else()
    add_subdirectory("${HYLE_MARL_DIR}" "${HYLE_BINARY_DIR}/_bench_deps/marl" EXCLUDE_FROM_ALL)
    set(_hyle_bench_marl_target marl)
  endif()
  add_executable(hyle_sched_marl_bench sync_marl_bench.cc)
  target_link_libraries(hyle_sched_marl_bench PRIVATE hyle_sched nanobench
                                                        ${_hyle_bench_marl_target})
  hyle_apply_warnings(hyle_sched_marl_bench)
  # sched-11 blocking_bench's marl comparison row (marl::blocking_call = new-thread-
  # per-call; the HYLE-pool-vs-marl-new-thread head-to-head). Same gate; links marl.
  if(TARGET hyle_sched_blocking_bench)
    target_compile_definitions(hyle_sched_blocking_bench PRIVATE HYLE_BENCH_MARL)
    target_link_libraries(hyle_sched_blocking_bench PRIVATE
      ${_hyle_bench_marl_target})
  endif()
  hyle_register_benches(sched hyle_sched_marl_bench)
endif()

hyle_register_benches(sched
  hyle_sched_fcontext_bench
  hyle_sched_sync_bench
  hyle_sched_sync_ext_bench
  hyle_sched_blocking_bench
  hyle_sched_cancel_bench
  hyle_sched_join_bench
  hyle_sched_ext_bench)
