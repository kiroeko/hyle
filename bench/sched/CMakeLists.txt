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

# sched-v7 dynamic TaskScope migration evidence. The peer is the natural
# pre-v7 public composition (spawn_task + same-allocator PMR handle storage +
# per-handle join/first-error aggregation), not a bare WaitGroup or detached
# route. Keep the pair opt-in and in separate executables so code placement and
# ownership state never leak across providers.
option(HYLE_SCHED_V7_TASK_SCOPE_P1_BENCH
  "build the Sched v7 dynamic TaskScope P1 pair" OFF)
if(HYLE_SCHED_V7_TASK_SCOPE_P1_BENCH)
  if(NOT WIN32 OR NOT HYLE_STATIC)
    message(FATAL_ERROR
      "HYLE_SCHED_V7_TASK_SCOPE_P1_BENCH requires Windows and HYLE_STATIC")
  endif()
  set(HYLE_SCHED_V7_BASELINE_DIR "" CACHE PATH
    "exact 37b62460 HYLE checkout for the pre-v7 manual route identity")
  if(NOT HYLE_SCHED_V7_BASELINE_DIR)
    message(FATAL_ERROR
      "HYLE_SCHED_V7_TASK_SCOPE_P1_BENCH requires HYLE_SCHED_V7_BASELINE_DIR")
  endif()
  set(_hyle_sched_v7_baseline_files
    include/hyle/sched/coro.h
    include/hyle/sched/join.h
    include/hyle/sched/scheduler.h
    include/hyle/sched/sync.h
    include/hyle/sched/waker.h
    include/hyle/mem/pmr.h
    src/sched/fiber.cc
    src/sched/scheduler.cc
    src/sched/sync.cc
    src/sched/join.cc
    src/sched/cancel.cc
    third_party/fcontext/asm/jump_x86_64_ms_pe_masm.asm
    third_party/fcontext/asm/make_x86_64_ms_pe_masm.asm
    third_party/fcontext/asm/ontop_x86_64_ms_pe_masm.asm)
  set(_hyle_sched_v7_baseline_hashes
    8694732bb1a069fde73f0359399b13f893d41c6981800f05cd824b293d93df54
    266d0bbc139304501d1501d915d6b45577e40ab63f8d043232a09f450ee8da86
    a97bb529d434980ead638fb20bb5ca8de324f2fb60e0fa698cf46bc018194f5f
    dfcc2c05e0205a22ce42634643aec65ccf85eda7fccae40c7a353fb6ed1d56d5
    cc89173f57dd37373899ad0955c37c203108fa2ff0932136f3451d657db2d0d7
    bd3c370b8729b5d82eeb62fa86798f020b8ffe22afdbaef023acbe7105e9f38f
    b327158f0f3e9d46a05a1da5aeb24f90651c0dc6453eda5beaa9aaf25f8ab3a8
    c6cdc451d904611ae5252372973a6bbb4d2324e64d406448c8b3f4a9394aed65
    564d5dc96e20ba9da6db1188455a360c005b6de8eeb75ed6d3e1501295f0572d
    b88776fc3828d123054d065da67f21c82069c93a5d90e6f7485a55eaa5a6c3d8
    634465e979b8978dc2c54a5d8b2c2e93ec302d2652f406d0aaf88aa98f559b8e
    27b88678004fccbab0a74cdca8a153641852a3036a410c05b714fc18cecc7469
    c14fa188661d1c0460fd660f221ff7f382af2e96b97e6a3932667f22b64f64a1
    ae05d63df36415f3bb6dd2188cf240dd7fc8c3bcf764290f485cef97561f6b0b)
  list(LENGTH _hyle_sched_v7_baseline_files _hyle_sched_v7_baseline_count)
  math(EXPR _hyle_sched_v7_baseline_last
    "${_hyle_sched_v7_baseline_count} - 1")
  foreach(_hyle_sched_v7_index RANGE 0 ${_hyle_sched_v7_baseline_last})
    list(GET _hyle_sched_v7_baseline_files ${_hyle_sched_v7_index}
      _hyle_sched_v7_file)
    list(GET _hyle_sched_v7_baseline_hashes ${_hyle_sched_v7_index}
      _hyle_sched_v7_expected)
    set(_hyle_sched_v7_path
      "${HYLE_SCHED_V7_BASELINE_DIR}/${_hyle_sched_v7_file}")
    if(NOT EXISTS "${_hyle_sched_v7_path}")
      message(FATAL_ERROR
        "Sched v7 baseline lacks ${_hyle_sched_v7_file}: ${HYLE_SCHED_V7_BASELINE_DIR}")
    endif()
    file(SHA256 "${_hyle_sched_v7_path}" _hyle_sched_v7_actual)
    if(NOT _hyle_sched_v7_actual STREQUAL _hyle_sched_v7_expected)
      message(FATAL_ERROR
        "Sched v7 37b62460 identity mismatch: ${_hyle_sched_v7_file}")
    endif()
    file(SHA256 "${HYLE_SOURCE_DIR}/${_hyle_sched_v7_file}"
      _hyle_sched_v7_candidate_actual)
    if(NOT _hyle_sched_v7_candidate_actual STREQUAL _hyle_sched_v7_expected)
      message(FATAL_ERROR
        "Sched v7 manual-route runtime drift: ${_hyle_sched_v7_file}")
    endif()
  endforeach()

  function(_hyle_configure_sched_v7_task_scope_bench target)
    target_link_libraries(${target} PRIVATE hyle_sched)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_sched_v7_task_scope_candidate_bench
    task_scope_p1_candidate_bench.cc)
  _hyle_configure_sched_v7_task_scope_bench(
    hyle_sched_v7_task_scope_candidate_bench)
  add_executable(hyle_sched_v7_task_scope_manual_bench
    task_scope_p1_manual_bench.cc)
  _hyle_configure_sched_v7_task_scope_bench(
    hyle_sched_v7_task_scope_manual_bench)
  add_executable(hyle_sched_v7_task_scope_candidate_placement_bench
    task_scope_p1_candidate_bench.cc)
  _hyle_configure_sched_v7_task_scope_bench(
    hyle_sched_v7_task_scope_candidate_placement_bench)
  add_executable(hyle_sched_v7_task_scope_manual_placement_bench
    task_scope_p1_manual_bench.cc)
  _hyle_configure_sched_v7_task_scope_bench(
    hyle_sched_v7_task_scope_manual_placement_bench)
  target_link_options(hyle_sched_v7_task_scope_candidate_placement_bench
    PRIVATE /FUNCTIONPADMIN:64)
  target_link_options(hyle_sched_v7_task_scope_manual_placement_bench
    PRIVATE /FUNCTIONPADMIN:64)

  hyle_register_benches(sched
    hyle_sched_v7_task_scope_candidate_bench
    hyle_sched_v7_task_scope_manual_bench
    hyle_sched_v7_task_scope_candidate_placement_bench
    hyle_sched_v7_task_scope_manual_placement_bench)
  unset(_hyle_sched_v7_baseline_files)
  unset(_hyle_sched_v7_baseline_hashes)
  unset(_hyle_sched_v7_baseline_count)
  unset(_hyle_sched_v7_baseline_last)
  unset(_hyle_sched_v7_index)
  unset(_hyle_sched_v7_file)
  unset(_hyle_sched_v7_expected)
  unset(_hyle_sched_v7_path)
  unset(_hyle_sched_v7_actual)
  unset(_hyle_sched_v7_candidate_actual)
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

# bench/peers owns the shared HYLE_BENCH_MARL checkout contract.
if(HYLE_BENCH_MARL)
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

  # sched-v1 P1 closure evidence. Keep candidate and peer in separate
  # executables so neither product changes the other's layout or link graph.
  # The shared instrument launches both under the same four-physical-core
  # policy and owns pairing, calibration and statistical verdicts.
  if(WIN32)
    add_executable(hyle_sched_p1_candidate_bench p1_candidate_bench.cc)
    target_link_libraries(hyle_sched_p1_candidate_bench PRIVATE hyle_sched)
    hyle_apply_warnings(hyle_sched_p1_candidate_bench)

    add_executable(hyle_sched_p1_peer_bench p1_peer_bench.cc)
    target_link_libraries(hyle_sched_p1_peer_bench PRIVATE
      ${_hyle_bench_marl_target})
    hyle_apply_warnings(hyle_sched_p1_peer_bench)

    # sched-v2 batch-structured P1 pair.  Keep it separate from the historical
    # rich schedule_task-vs-batch row so old raw evidence stays reproducible.
    add_executable(hyle_sched_bulk_p1_candidate_marl_bench
      bulk_p1_candidate_bench.cc)
    target_link_libraries(hyle_sched_bulk_p1_candidate_marl_bench PRIVATE
      hyle_sched)
    hyle_apply_warnings(hyle_sched_bulk_p1_candidate_marl_bench)

    add_executable(hyle_sched_bulk_p1_marl_bench bulk_p1_marl_bench.cc)
    target_link_libraries(hyle_sched_bulk_p1_marl_bench PRIVATE
      ${_hyle_bench_marl_target})
    hyle_apply_warnings(hyle_sched_bulk_p1_marl_bench)
  endif()
  # sched-11 blocking_bench's marl comparison row (marl::blocking_call = new-thread-
  # per-call; the HYLE-pool-vs-marl-new-thread head-to-head). Same gate; links marl.
  if(TARGET hyle_sched_blocking_bench)
    target_compile_definitions(hyle_sched_blocking_bench PRIVATE HYLE_BENCH_MARL)
    target_link_libraries(hyle_sched_blocking_bench PRIVATE
      ${_hyle_bench_marl_target})
  endif()
  hyle_register_benches(sched hyle_sched_marl_bench)
  if(TARGET hyle_sched_p1_candidate_bench)
    hyle_register_benches(sched
      hyle_sched_p1_candidate_bench hyle_sched_p1_peer_bench)
  endif()
  if(TARGET hyle_sched_bulk_p1_candidate_marl_bench)
    hyle_register_benches(sched
      hyle_sched_bulk_p1_candidate_marl_bench
      hyle_sched_bulk_p1_marl_bench)
  endif()
endif()

# oneTBB is the second mature same-contract batch peer.  Its synchronous
# parallel_for route and HYLE bulk_execute live in separate processes so neither
# provider changes the other's code placement or runtime initialization.
if(WIN32 AND HYLE_BENCH_TBB)
  add_executable(hyle_sched_bulk_p1_candidate_tbb_bench
    bulk_p1_candidate_bench.cc)
  target_link_libraries(hyle_sched_bulk_p1_candidate_tbb_bench PRIVATE
    hyle_sched)
  hyle_apply_warnings(hyle_sched_bulk_p1_candidate_tbb_bench)

  add_executable(hyle_sched_bulk_p1_tbb_bench bulk_p1_tbb_bench.cc)
  target_link_libraries(hyle_sched_bulk_p1_tbb_bench PRIVATE TBB::tbb)
  hyle_apply_warnings(hyle_sched_bulk_p1_tbb_bench)
  hyle_stage_tbb_benchmark_runtime(hyle_sched_bulk_p1_tbb_bench)

  hyle_register_benches(sched
    hyle_sched_bulk_p1_candidate_tbb_bench
    hyle_sched_bulk_p1_tbb_bench)
endif()

hyle_register_benches(sched
  hyle_sched_fcontext_bench
  hyle_sched_sync_bench
  hyle_sched_sync_ext_bench
  hyle_sched_blocking_bench
  hyle_sched_cancel_bench
  hyle_sched_join_bench
  hyle_sched_ext_bench)

# sched-v6 pinned structured-task migration evidence. The peer is the exact
# pre-v6 public composition (Task + spawn_detached_on + typed slot + per-task
# WaitGroup), not a weaker unpinned/detached/non-suspending workload. Keep this
# opt-in so ordinary benchmark builds do not pay for closure-only artifacts.
option(HYLE_SCHED_V6_PINNED_TASK_P1_BENCH
  "build the Sched v6 pinned structured-task P1 pair" OFF)
if(HYLE_SCHED_V6_PINNED_TASK_P1_BENCH)
  if(NOT WIN32 OR NOT HYLE_STATIC)
    message(FATAL_ERROR
      "HYLE_SCHED_V6_PINNED_TASK_P1_BENCH requires Windows and HYLE_STATIC")
  endif()
  set(HYLE_SCHED_V6_BASELINE_DIR "" CACHE PATH
    "exact cb87f502 HYLE checkout for the pre-v6 manual route identity")
  if(NOT HYLE_SCHED_V6_BASELINE_DIR)
    message(FATAL_ERROR
      "HYLE_SCHED_V6_PINNED_TASK_P1_BENCH requires HYLE_SCHED_V6_BASELINE_DIR")
  endif()
  set(_hyle_sched_v6_baseline_files
    include/hyle/sched/join.h
    include/hyle/sched/coro.h
    include/hyle/sched/fiber.h
    include/hyle/sched/scheduler.h
    include/hyle/sched/sync.h
    include/hyle/sched/waker.h
    src/sched/fiber.cc
    src/sched/scheduler.cc
    src/sched/sync.cc
    third_party/fcontext/asm/jump_x86_64_ms_pe_masm.asm
    third_party/fcontext/asm/make_x86_64_ms_pe_masm.asm
    third_party/fcontext/asm/ontop_x86_64_ms_pe_masm.asm)
  set(_hyle_sched_v6_baseline_hashes
    633cf941c480fed70054ecfa559dedd3e5eb1c7a02c758c3f1e3e250039c9687
    8694732bb1a069fde73f0359399b13f893d41c6981800f05cd824b293d93df54
    bc7d0a8b7d58fbcb14843b27f6c4f910ba18be9e50cc7f0c29d6cc59d57aa278
    a97bb529d434980ead638fb20bb5ca8de324f2fb60e0fa698cf46bc018194f5f
    dfcc2c05e0205a22ce42634643aec65ccf85eda7fccae40c7a353fb6ed1d56d5
    cc89173f57dd37373899ad0955c37c203108fa2ff0932136f3451d657db2d0d7
    b327158f0f3e9d46a05a1da5aeb24f90651c0dc6453eda5beaa9aaf25f8ab3a8
    c6cdc451d904611ae5252372973a6bbb4d2324e64d406448c8b3f4a9394aed65
    564d5dc96e20ba9da6db1188455a360c005b6de8eeb75ed6d3e1501295f0572d
    27b88678004fccbab0a74cdca8a153641852a3036a410c05b714fc18cecc7469
    c14fa188661d1c0460fd660f221ff7f382af2e96b97e6a3932667f22b64f64a1
    ae05d63df36415f3bb6dd2188cf240dd7fc8c3bcf764290f485cef97561f6b0b)
  list(LENGTH _hyle_sched_v6_baseline_files _hyle_sched_v6_baseline_count)
  math(EXPR _hyle_sched_v6_baseline_last
    "${_hyle_sched_v6_baseline_count} - 1")
  foreach(_hyle_sched_v6_index RANGE 0 ${_hyle_sched_v6_baseline_last})
    list(GET _hyle_sched_v6_baseline_files ${_hyle_sched_v6_index}
      _hyle_sched_v6_file)
    list(GET _hyle_sched_v6_baseline_hashes ${_hyle_sched_v6_index}
      _hyle_sched_v6_expected)
    set(_hyle_sched_v6_path
      "${HYLE_SCHED_V6_BASELINE_DIR}/${_hyle_sched_v6_file}")
    if(NOT EXISTS "${_hyle_sched_v6_path}")
      message(FATAL_ERROR
        "Sched v6 baseline lacks ${_hyle_sched_v6_file}: ${HYLE_SCHED_V6_BASELINE_DIR}")
    endif()
    file(SHA256 "${_hyle_sched_v6_path}" _hyle_sched_v6_actual)
    if(NOT _hyle_sched_v6_actual STREQUAL _hyle_sched_v6_expected)
      message(FATAL_ERROR
        "Sched v6 cb87f502 identity mismatch: ${_hyle_sched_v6_file}")
    endif()
    # The manual peer is linked against the candidate build only because every
    # runtime file it executes is byte-identical to cb87f502. join.h is the sole
    # intentional exception: the candidate adds spawn_task_on there, while the
    # peer source does not include or call it.
    if(NOT _hyle_sched_v6_file STREQUAL "include/hyle/sched/join.h")
      file(SHA256 "${HYLE_SOURCE_DIR}/${_hyle_sched_v6_file}"
        _hyle_sched_v6_candidate_actual)
      if(NOT _hyle_sched_v6_candidate_actual STREQUAL
         _hyle_sched_v6_expected)
        message(FATAL_ERROR
          "Sched v6 manual-route runtime drift: ${_hyle_sched_v6_file}")
      endif()
    endif()
  endforeach()

  function(_hyle_configure_sched_v6_pinned_bench target)
    target_link_libraries(${target} PRIVATE hyle_sched)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_sched_v6_pinned_candidate_bench
    pinned_task_p1_candidate_bench.cc)
  _hyle_configure_sched_v6_pinned_bench(
    hyle_sched_v6_pinned_candidate_bench)
  add_executable(hyle_sched_v6_pinned_manual_bench
    pinned_task_p1_manual_bench.cc)
  _hyle_configure_sched_v6_pinned_bench(
    hyle_sched_v6_pinned_manual_bench)

  add_executable(hyle_sched_v6_pinned_candidate_placement_bench
    pinned_task_p1_candidate_bench.cc)
  _hyle_configure_sched_v6_pinned_bench(
    hyle_sched_v6_pinned_candidate_placement_bench)
  add_executable(hyle_sched_v6_pinned_manual_placement_bench
    pinned_task_p1_manual_bench.cc)
  _hyle_configure_sched_v6_pinned_bench(
    hyle_sched_v6_pinned_manual_placement_bench)
  target_link_options(hyle_sched_v6_pinned_candidate_placement_bench PRIVATE
    /FUNCTIONPADMIN:64)
  target_link_options(hyle_sched_v6_pinned_manual_placement_bench PRIVATE
    /FUNCTIONPADMIN:64)

  hyle_register_benches(sched
    hyle_sched_v6_pinned_candidate_bench
    hyle_sched_v6_pinned_manual_bench
    hyle_sched_v6_pinned_candidate_placement_bench
    hyle_sched_v6_pinned_manual_placement_bench)
  unset(_hyle_sched_v6_baseline_files)
  unset(_hyle_sched_v6_baseline_hashes)
  unset(_hyle_sched_v6_baseline_count)
  unset(_hyle_sched_v6_baseline_last)
  unset(_hyle_sched_v6_index)
  unset(_hyle_sched_v6_file)
  unset(_hyle_sched_v6_expected)
  unset(_hyle_sched_v6_path)
  unset(_hyle_sched_v6_actual)
  unset(_hyle_sched_v6_candidate_actual)
endif()
