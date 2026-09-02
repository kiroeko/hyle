hyle_add_bench(mem SOURCES allocator_bench.cc DEPS mimalloc_static)

# arena/pool/stack specialized-allocator cluster vs std::pmr (std's arena/pool, no
# external clone) + raw mimalloc. Own exe (allocator_bench.cc already owns main()).
add_executable(hyle_mem_scenarios_bench scenario_bench.cc)
target_link_libraries(hyle_mem_scenarios_bench PRIVATE hyle_mem nanobench
                                                       mimalloc_static)
if(WIN32)
  target_link_libraries(hyle_mem_scenarios_bench PRIVATE psapi)
endif()
hyle_apply_warnings(hyle_mem_scenarios_bench)

# FiberStackPool is a scheduler-owned mem specialization. Certify it only
# through the real Scheduler + Fiber create/run/destroy lifecycle, with direct
# guarded VM reserve/release as the same-lifecycle baseline.
if(HYLE_HAS_FCONTEXT)
  add_executable(hyle_mem_fiber_stack_bench fiber_stack_bench.cc)
  target_link_libraries(hyle_mem_fiber_stack_bench PRIVATE hyle_sched)
  hyle_apply_warnings(hyle_mem_fiber_stack_bench)
endif()

if(HYLE_BENCH_TBB)
  target_compile_definitions(hyle_mem_scenarios_bench PRIVATE
                             HYLE_BENCH_TBB TBB_PREVIEW_MEMORY_POOL=1)
  target_link_libraries(hyle_mem_scenarios_bench PRIVATE TBB::tbbmalloc
                                                        TBB::tbb)
endif()

hyle_register_benches(mem
  hyle_mem_scenarios_bench)
if(TARGET hyle_mem_fiber_stack_bench)
  hyle_register_benches(mem hyle_mem_fiber_stack_bench)
endif()

# mem-v2 closure evidence.  This opt-in adapter keeps the four admitted hot
# families in one source-consumer artifact: isolated Heap allocation, external
# scratch epochs, BufferView range dispatch, and warmed transient BufferPool
# reuse.  Mapping/residency and LifetimePlan construction remain P0
# complexity/resource families and deliberately do not acquire timing rows.
option(HYLE_MEM_V2_P1_BENCH
  "build the Mem v2 source-consumer evidence adapter" OFF)
if(HYLE_MEM_V2_P1_BENCH)
  if(NOT HYLE_STATIC OR NOT HYLE_BENCH_TBB)
    message(FATAL_ERROR
      "HYLE_MEM_V2_P1_BENCH requires HYLE_STATIC and the oneTBB peer")
  endif()

  set(HYLE_MEM_V2_BASELINE_DIR "${HYLE_SOURCE_DIR}" CACHE PATH
      "locked HYLE snapshot containing the Mem v2 direct/lower-bound routes")
  foreach(_hyle_mem_v2_baseline_file IN ITEMS
      third_party/mimalloc/include/mimalloc.h
      include/hyle/mem/heap.h
      include/hyle/mem/region.h
      include/hyle/mem/buffer.h
      include/hyle/mem/buffer_pool.h
      src/mem/heap.cc
      src/mem/resource.cc
      src/mem/buffer_pool.cc)
    if(NOT EXISTS
        "${HYLE_MEM_V2_BASELINE_DIR}/${_hyle_mem_v2_baseline_file}")
      message(FATAL_ERROR
        "Mem v2 baseline lacks ${_hyle_mem_v2_baseline_file}: "
        "${HYLE_MEM_V2_BASELINE_DIR}")
    endif()
    file(SHA256 "${HYLE_SOURCE_DIR}/${_hyle_mem_v2_baseline_file}"
      _hyle_mem_v2_candidate_sha256)
    file(SHA256
      "${HYLE_MEM_V2_BASELINE_DIR}/${_hyle_mem_v2_baseline_file}"
      _hyle_mem_v2_baseline_sha256)
    if(NOT _hyle_mem_v2_candidate_sha256 STREQUAL
       _hyle_mem_v2_baseline_sha256)
      message(FATAL_ERROR
        "Mem v2 lower-bound source drift: ${_hyle_mem_v2_baseline_file}")
    endif()
  endforeach()
  unset(_hyle_mem_v2_baseline_file)
  unset(_hyle_mem_v2_candidate_sha256)
  unset(_hyle_mem_v2_baseline_sha256)

  function(_hyle_configure_mem_v2_consumer_bench target)
    target_link_libraries(${target} PRIVATE
      hyle_mem
      nanobench
      mimalloc_static
      TBB::tbbmalloc
      TBB::tbb)
    target_compile_definitions(${target} PRIVATE
      TBB_PREVIEW_MEMORY_POOL=1)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_stage_tbb_benchmark_runtime(${target})
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_mem_v2_consumer_bench v2_consumer_bench.cc)
  _hyle_configure_mem_v2_consumer_bench(hyle_mem_v2_consumer_bench)

  add_executable(hyle_mem_v2_consumer_bench_placement v2_consumer_bench.cc)
  _hyle_configure_mem_v2_consumer_bench(
    hyle_mem_v2_consumer_bench_placement)
  if(MSVC)
    target_link_options(hyle_mem_v2_consumer_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_mem_v2_consumer_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "Mem v2 consumer placement perturbation is unsupported")
  endif()

  hyle_register_benches(mem
    hyle_mem_v2_consumer_bench
    hyle_mem_v2_consumer_bench_placement)
endif()

# mem-v1 P1 owner/consumer contract. The benchmark charges complete consumer
# batches rather than sub-nanosecond allocator calls: runtime allocator
# injection, request Arena epochs and scheduler-owned guarded-stack lifecycle.
# Boost PMR and oneTBB are exact external peers; the direct system/fiber routes
# are the compile-time/cold baselines of the same candidate implementation.
if(HYLE_STATIC AND HYLE_HAS_FCONTEXT AND HYLE_BENCH_TBB AND HYLE_BENCH_BOOST)
  # The direct allocator and guarded-VM rows are same-product controls, not an
  # external SOTA peer. Qualification binds them to the last product commit by
  # passing a clean HYLE worktree here; manual builds compare against themselves.
  # Byte equality prevents a later implementation drift from masquerading as
  # the frozen direct baseline while still compiling both routes into one fair
  # artifact/link layout.
  set(HYLE_MEM_BASELINE_DIR "${HYLE_SOURCE_DIR}" CACHE PATH
      "HYLE product snapshot used by mem direct-baseline controls")
  foreach(_hyle_mem_baseline_file IN ITEMS
      include/hyle/mem/allocator.h
      include/hyle/mem/fiber_stack.h
      include/hyle/mem/system_inline.h
      include/hyle/sched/fiber.h
      include/hyle/sched/scheduler.h
      src/mem/system.cc
      src/sched/fiber.cc
      src/sched/scheduler.cc)
    if(NOT EXISTS
        "${HYLE_MEM_BASELINE_DIR}/${_hyle_mem_baseline_file}")
      message(FATAL_ERROR
        "mem baseline lacks ${_hyle_mem_baseline_file}: "
        "${HYLE_MEM_BASELINE_DIR}")
    endif()
    file(SHA256 "${HYLE_SOURCE_DIR}/${_hyle_mem_baseline_file}"
      _hyle_mem_candidate_sha256)
    file(SHA256
      "${HYLE_MEM_BASELINE_DIR}/${_hyle_mem_baseline_file}"
      _hyle_mem_baseline_sha256)
    if(NOT _hyle_mem_candidate_sha256 STREQUAL _hyle_mem_baseline_sha256)
      message(FATAL_ERROR
        "mem direct-baseline source drift: ${_hyle_mem_baseline_file}")
    endif()
  endforeach()
  unset(_hyle_mem_baseline_file)
  unset(_hyle_mem_candidate_sha256)
  unset(_hyle_mem_baseline_sha256)

  add_library(hyle_mem_boost_pmr_peer STATIC
    "${HYLE_BOOST_DIR}/libs/container/src/global_resource.cpp"
    "${HYLE_BOOST_DIR}/libs/container/src/monotonic_buffer_resource.cpp")
  target_link_libraries(hyle_mem_boost_pmr_peer
    PUBLIC hyle_bench_boost_headers)
  target_compile_definitions(hyle_mem_boost_pmr_peer PRIVATE
    BOOST_ALL_NO_LIB=1)
  if(MSVC)
    target_compile_options(hyle_mem_boost_pmr_peer PRIVATE /wd4100 /wd4127)
  endif()

  function(_hyle_configure_mem_owner_consumer target)
    target_link_libraries(${target} PRIVATE hyle_sched nanobench
      TBB::tbbmalloc TBB::tbb hyle_mem_boost_pmr_peer
      hyle_bench_boost_headers)
    target_compile_definitions(${target} PRIVATE BOOST_ALL_NO_LIB=1)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_stage_tbb_benchmark_runtime(${target})
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_mem_owner_consumer_bench
    owner_consumer_bench.cc tbb_runtime_peer.cc)
  _hyle_configure_mem_owner_consumer(hyle_mem_owner_consumer_bench)

  add_executable(hyle_mem_owner_consumer_bench_placement
    owner_consumer_bench.cc tbb_runtime_peer.cc)
  _hyle_configure_mem_owner_consumer(
    hyle_mem_owner_consumer_bench_placement)
  if(MSVC)
    target_link_options(hyle_mem_owner_consumer_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_mem_owner_consumer_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "mem owner consumer placement perturbation is unsupported")
  endif()
  hyle_register_benches(mem
    hyle_mem_owner_consumer_bench
    hyle_mem_owner_consumer_bench_placement)
endif()

# Windows is the registered cross-core engineering machine. Build Marl's exact
# source list locally because its top-level project declares generic ASM, which
# modern MSVC correctly refuses to treat cl.exe as. The published pool
# algorithm is used through marl::BoundedPool with the audited compatibility
# overlay below.
if(HYLE_STATIC AND HYLE_BENCH_MARL AND MSVC)
  # The pinned Marl header declares a non-blocking tryBorrow() but contains one
  # stale assignment to Item::pool, a member removed from the current Loan
  # design.  Generate a build-local, one-line compatibility overlay so the
  # benchmark can exercise Marl's published non-blocking capacity contract.
  # The external checkout remains clean and its algorithm is otherwise exact.
  file(READ "${HYLE_MARL_DIR}/include/marl/pool.h"
    _hyle_marl_pool_header)
  string(REPLACE "    item->pool = this;\n" ""
    _hyle_marl_pool_patched "${_hyle_marl_pool_header}")
  if(_hyle_marl_pool_header STREQUAL _hyle_marl_pool_patched)
    message(FATAL_ERROR
      "Pinned Marl pool compatibility patch no longer matches; re-audit peer")
  endif()
  # The overlay lives in a build directory rather than beside Marl's sibling
  # headers, so spell those three otherwise-relative includes through the
  # original public include root.
  string(REPLACE "\"conditionvariable.h\"" "<marl/conditionvariable.h>"
    _hyle_marl_pool_patched "${_hyle_marl_pool_patched}")
  string(REPLACE "\"memory.h\"" "<marl/memory.h>"
    _hyle_marl_pool_patched "${_hyle_marl_pool_patched}")
  string(REPLACE "\"mutex.h\"" "<marl/mutex.h>"
    _hyle_marl_pool_patched "${_hyle_marl_pool_patched}")
  set(_hyle_marl_overlay_dir
    "${CMAKE_CURRENT_BINARY_DIR}/marl-compat/include")
  file(MAKE_DIRECTORY "${_hyle_marl_overlay_dir}/marl")
  file(WRITE "${_hyle_marl_overlay_dir}/marl/pool.h"
    "${_hyle_marl_pool_patched}")

  add_library(hyle_mem_marl_peer STATIC
    "${HYLE_MARL_DIR}/src/debug.cpp"
    "${HYLE_MARL_DIR}/src/memory.cpp"
    "${HYLE_MARL_DIR}/src/scheduler.cpp"
    "${HYLE_MARL_DIR}/src/thread.cpp"
    "${HYLE_MARL_DIR}/src/trace.cpp")
  target_include_directories(hyle_mem_marl_peer SYSTEM BEFORE PUBLIC
    "${_hyle_marl_overlay_dir}"
    "${HYLE_MARL_DIR}/include")
  target_compile_definitions(hyle_mem_marl_peer PRIVATE
    _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(hyle_mem_marl_peer PRIVATE /wd4127 /wd4324)
  target_link_libraries(hyle_mem_marl_peer PUBLIC Threads::Threads)

  function(_hyle_configure_mem_handoff_consumer target)
    target_link_libraries(${target} PRIVATE hyle_mem nanobench
      hyle_mem_marl_peer)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_mem_handoff_consumer_bench
    handoff_consumer_bench.cc)
  _hyle_configure_mem_handoff_consumer(
    hyle_mem_handoff_consumer_bench)

  add_executable(hyle_mem_handoff_consumer_bench_placement
    handoff_consumer_bench.cc)
  _hyle_configure_mem_handoff_consumer(
    hyle_mem_handoff_consumer_bench_placement)
  target_link_options(hyle_mem_handoff_consumer_bench_placement PRIVATE
    /FUNCTIONPADMIN:64)
  hyle_register_benches(mem
    hyle_mem_handoff_consumer_bench
    hyle_mem_handoff_consumer_bench_placement)

  unset(_hyle_marl_overlay_dir)
  unset(_hyle_marl_pool_header)
  unset(_hyle_marl_pool_patched)
endif()
