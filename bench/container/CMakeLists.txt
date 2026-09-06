hyle_add_bench(container SOURCES container_bench_main.cc flat_hash_map_bench.cc
  string_buf_bench.cc string_view_bench.cc path_bench.cc string_algo_bench.cc
  DEPS hyle_text hyle_platform)
# string_algo utf8 bench races simdutf (the SIMD utf8 SOTA) for the comparison row.
# simdutf is VENDORED in third_party and compiled into hyle_text (the product uses it
# unconditionally), but those symbols are PRIVATE/non-exported across the hyle_text DLL
# boundary -- the bench can't borrow them transitively. So compile simdutf.cpp into the bench
# exe too: it's a SEPARATE binary from the DLL, so its own copy is NOT a duplicate symbol (the
# DLL exports none of them). The gate just toggles the bench-side comparison row.
option(HYLE_BENCH_SIMDUTF "build string_algo utf8 bench vs simdutf (SIMD SOTA)" OFF)
if(HYLE_BENCH_SIMDUTF)
  set(_simdutf "${PROJECT_SOURCE_DIR}/third_party/simdutf")
  target_compile_definitions(hyle_container_bench PRIVATE HYLE_BENCH_SIMDUTF)
  target_include_directories(hyle_container_bench SYSTEM PRIVATE "${_simdutf}")
  target_sources(hyle_container_bench PRIVATE "${_simdutf}/simdutf.cpp")
  # Same -Wall/-Werror exemption as the product build (src/text): the amalgamation's
  # own -Wpedantic warnings would break a HYLE_WERROR=ON bench build.
  if(MSVC)
    set_source_files_properties("${_simdutf}/simdutf.cpp" PROPERTIES COMPILE_OPTIONS "/w")
  else()
    set_source_files_properties("${_simdutf}/simdutf.cpp" PROPERTIES COMPILE_OPTIONS "-w")
  endif()
endif()

add_executable(hyle_queue_sota_bench queue_sota_bench.cc)
# BoundedQueue/ShardedQueue default to compile-time SystemAllocator. A shared
# build reaches the private backend through hyle_mem; a static build inherits
# mimalloc as hyle_mem's link-only implementation dependency.
target_link_libraries(hyle_queue_sota_bench PRIVATE hyle_container_queue nanobench)
hyle_apply_warnings(hyle_queue_sota_bench)

# task #19 breadth + gap-fill #34/#35: the claim-bearing containers. SmallVector
# (inline storage) and PodVector (POD dynamic array) vs std::vector. The real
# SmallVector peer is opt-in absl::InlinedVector (-DHYLE_BENCH_ABSL, wired in
# the absl block below). SmallVector
# defaults to the compile-time SystemAllocator and must close through hyle_mem,
# not a benchmark-only vendor link.
add_executable(hyle_container_misc_bench container_misc_bench.cc)
target_link_libraries(hyle_container_misc_bench PRIVATE hyle_container hyle_mem nanobench)
hyle_apply_warnings(hyle_container_misc_bench)
add_executable(hyle_container_set_bench set_bench.cc)
target_link_libraries(hyle_container_set_bench PRIVATE hyle_container nanobench)
hyle_apply_warnings(hyle_container_set_bench)
add_executable(hyle_container_ring_bench ring_bench.cc)
target_link_libraries(hyle_container_ring_bench PRIVATE hyle_container nanobench)
hyle_apply_warnings(hyle_container_ring_bench)
add_executable(hyle_container_spsc_bench spsc_bench.cc)
target_link_libraries(hyle_container_spsc_bench PRIVATE hyle_container_queue nanobench)
hyle_apply_warnings(hyle_container_spsc_bench)
add_executable(hyle_container_concurrent_map_bench concurrent_map_bench.cc)
target_link_libraries(hyle_container_concurrent_map_bench PRIVATE hyle_container_concurrent nanobench)
hyle_apply_warnings(hyle_container_concurrent_map_bench)
add_executable(hyle_container_flat_sorted_bench flat_sorted_bench.cc)
target_link_libraries(hyle_container_flat_sorted_bench PRIVATE hyle_container nanobench)
hyle_apply_warnings(hyle_container_flat_sorted_bench)

# task #23: hash cluster. HYLE hash_bytes(XXH3)/mix64 vs raw XXH3 (vendored, always)
# + std::hash. wyhash is a sibling clone -> opt-in -DHYLE_BENCH_WYHASH=ON (header-only).
add_executable(hyle_hash_bench hash_bench.cc)
target_link_libraries(hyle_hash_bench PRIVATE hyle_container hyle_platform nanobench)
target_include_directories(hyle_hash_bench SYSTEM PRIVATE "${PROJECT_SOURCE_DIR}/third_party/xxhash")
hyle_apply_warnings(hyle_hash_bench)
if(HYLE_TARGET_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  target_sources(hyle_hash_bench PRIVATE
    hash_xxh3_avx2_peer.cc hash_xxh3_avx512_peer.cc)
  target_compile_definitions(hyle_hash_bench PRIVATE
    HYLE_HASH_PEER_AVX2 HYLE_HASH_PEER_AVX512)
  hyle_set_isa(hash_xxh3_avx2_peer.cc AVX2)
  hyle_set_isa(hash_xxh3_avx512_peer.cc AVX512)
endif()
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

if(HYLE_BENCH_TBB)
  if(TARGET hyle_queue_sota_bench)
    target_compile_definitions(hyle_queue_sota_bench PRIVATE HYLE_BENCH_TBB)
    target_link_libraries(hyle_queue_sota_bench PRIVATE TBB::tbb)
  endif()
  if(TARGET hyle_container_concurrent_map_bench)
    target_compile_definitions(hyle_container_concurrent_map_bench PRIVATE
                               HYLE_BENCH_TBB)
    target_link_libraries(hyle_container_concurrent_map_bench PRIVATE TBB::tbb)
  endif()
endif()

if(HYLE_BENCH_ABSL)
  add_executable(hyle_absl_bench flat_hash_vs_absl_bench.cc)
  target_link_libraries(hyle_absl_bench PRIVATE hyle_container nanobench
    absl::flat_hash_map absl::flat_hash_set)
  hyle_mark_absl_benchmark_consumer(hyle_absl_bench)
  hyle_apply_warnings(hyle_absl_bench)
  add_executable(hyle_container_btree_bench
    btree_vs_absl_bench.cc)
  target_link_libraries(hyle_container_btree_bench PRIVATE hyle_container
    nanobench absl::btree)
  hyle_mark_absl_benchmark_consumer(hyle_container_btree_bench)
  hyle_apply_warnings(hyle_container_btree_bench)
  # absl::InlinedVector is the small-vector SOTA -> feed it to the misc bench too.
  target_compile_definitions(hyle_container_misc_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_container_misc_bench PRIVATE absl::inlined_vector)
  hyle_mark_absl_benchmark_consumer(hyle_container_misc_bench)
  # StringInterner's strongest same-contract peer is a sharded, locked
  # absl::flat_hash_set interner (not an unlocked std map floor).
  target_compile_definitions(hyle_container_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_container_bench PRIVATE absl::flat_hash_set)
  hyle_mark_absl_benchmark_consumer(hyle_container_bench)
  hyle_register_benches(container
    hyle_absl_bench
    hyle_container_btree_bench)
endif()

hyle_register_benches(container
    hyle_queue_sota_bench
    hyle_container_misc_bench
    hyle_container_set_bench
    hyle_container_ring_bench
    hyle_hash_bench
    hyle_container_concurrent_map_bench)
hyle_register_benches(container
  hyle_container_spsc_bench
  hyle_container_flat_sorted_bench)

# Container v5 product evidence is intentionally narrow: one normal workload
# for each reviewed internal factor.  Keep the single-thread and structured
# multi-worker adapters independently selectable so a qualification contract
# does not acquire an unused peer dependency.
option(HYLE_CONTAINER_V5_SINGLE_P1_BENCH
  "build the Container v5 LRU and UTF product evidence adapter" OFF)
if(HYLE_CONTAINER_V5_SINGLE_P1_BENCH)
  if(NOT HYLE_STATIC OR NOT HYLE_BENCH_ABSL)
    message(FATAL_ERROR
      "HYLE_CONTAINER_V5_SINGLE_P1_BENCH requires HYLE_STATIC plus the "
      "Abseil benchmark peer")
  endif()
  set(HYLE_SIMDUTF_PEER_DIR "${HYLE_SOURCE_DIR}/../simdutf" CACHE PATH
      "source-pinned simdutf checkout for Container v5 evidence")
  if(NOT EXISTS "${HYLE_SIMDUTF_PEER_DIR}/include/simdutf.h")
    message(FATAL_ERROR
      "HYLE_CONTAINER_V5_SINGLE_P1_BENCH requires a source-pinned simdutf "
      "checkout at HYLE_SIMDUTF_PEER_DIR")
  endif()

  function(_hyle_configure_container_v5_single target)
    target_link_libraries(${target} PRIVATE
      hyle_text hyle_container hyle_mem nanobench absl::flat_hash_map)
    target_include_directories(${target} SYSTEM BEFORE PRIVATE
      "${HYLE_SIMDUTF_PEER_DIR}/include")
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_mark_absl_benchmark_consumer(${target})
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_container_v5_p1_bench product_v5_bench.cc)
  _hyle_configure_container_v5_single(hyle_container_v5_p1_bench)
  add_executable(hyle_container_v5_p1_bench_placement product_v5_bench.cc)
  _hyle_configure_container_v5_single(
    hyle_container_v5_p1_bench_placement)
  if(MSVC)
    target_link_options(hyle_container_v5_p1_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_container_v5_p1_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR "Container v5 placement perturbation is unsupported")
  endif()

  hyle_register_benches(container
    hyle_container_v5_p1_bench
    hyle_container_v5_p1_bench_placement)
endif()

option(HYLE_CONTAINER_V5_CONCURRENT_P1_BENCH
  "build the Container v5 concurrent set and queue evidence adapters" OFF)
if(HYLE_CONTAINER_V5_CONCURRENT_P1_BENCH)
  if(NOT HYLE_STATIC OR NOT HYLE_BENCH_TBB)
    message(FATAL_ERROR
      "HYLE_CONTAINER_V5_CONCURRENT_P1_BENCH requires HYLE_STATIC plus the "
      "oneTBB benchmark peer")
  endif()

  add_executable(hyle_container_v5_concurrent_bench
    product_v5_concurrent_bench.cc)
  target_link_libraries(hyle_container_v5_concurrent_bench PRIVATE
    hyle_container_concurrent hyle_container_queue Threads::Threads)
  set_property(TARGET hyle_container_v5_concurrent_bench
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  hyle_apply_warnings(hyle_container_v5_concurrent_bench)

  add_executable(hyle_container_v5_concurrent_peer_bench
    product_v5_concurrent_bench.cc)
  target_compile_definitions(hyle_container_v5_concurrent_peer_bench PRIVATE
    HYLE_CONTAINER_V5_PEER)
  target_link_libraries(hyle_container_v5_concurrent_peer_bench PRIVATE
    TBB::tbb Threads::Threads)
  set_property(TARGET hyle_container_v5_concurrent_peer_bench
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  hyle_apply_warnings(hyle_container_v5_concurrent_peer_bench)
  hyle_stage_tbb_benchmark_runtime(
    hyle_container_v5_concurrent_peer_bench)

  hyle_register_benches(container
    hyle_container_v5_concurrent_bench
    hyle_container_v5_concurrent_peer_bench)
endif()

# Historical container-v2 closure-perf-v1 adapter.  Keep it buildable for its
# immutable receipt, but do not reuse its five mixed rows as current-method
# consumer evidence.  It is opt-in because all four source-pinned peers must be
# present and clean for the unified runner.
option(HYLE_CONTAINER_P1_BENCH
  "build the historical Container v2 five-row evidence adapter" OFF)
if(HYLE_CONTAINER_P1_BENCH)
  if(NOT HYLE_STATIC OR NOT HYLE_BENCH_ABSL OR NOT HYLE_BENCH_BOOST OR
     NOT HYLE_BENCH_RIGTORP)
    message(FATAL_ERROR
      "HYLE_CONTAINER_P1_BENCH requires HYLE_STATIC plus the Abseil, Boost, "
      "and rigtorp benchmark peers")
  endif()

  set(HYLE_XXHASH_PEER_DIR "${PROJECT_SOURCE_DIR}/../xxHash" CACHE PATH
      "source-pinned upstream xxHash checkout for container P1")
  if(NOT EXISTS "${HYLE_XXHASH_PEER_DIR}/xxhash.h")
    message(FATAL_ERROR
      "HYLE_CONTAINER_P1_BENCH requires xxhash.h in "
      "${HYLE_XXHASH_PEER_DIR}")
  endif()
  if(NOT EXISTS
      "${HYLE_BOOST_DIR}/libs/dynamic_bitset/include/boost/dynamic_bitset.hpp")
    message(FATAL_ERROR
      "HYLE_CONTAINER_P1_BENCH requires Boost.DynamicBitset in "
      "${HYLE_BOOST_DIR}")
  endif()

  function(_hyle_configure_container_p1_bench target)
    target_link_libraries(${target} PRIVATE
      hyle_container
      nanobench
      hyle_bench_boost_headers
      absl::flat_hash_map
      absl::inlined_vector)
    target_include_directories(${target} SYSTEM PRIVATE
      "${HYLE_BOOST_DIR}/libs/dynamic_bitset/include"
      "${HYLE_BOOST_DIR}/libs/container_hash/include"
      "${HYLE_BOOST_DIR}/libs/describe/include"
      "${HYLE_BOOST_DIR}/libs/mp11/include"
      "${HYLE_BOOST_DIR}/libs/throw_exception/include"
      "${HYLE_BOOST_DIR}/libs/type_traits/include"
      "${HYLE_RIGTORP_DIR}/include")
    target_include_directories(${target} SYSTEM BEFORE PRIVATE
      "${HYLE_XXHASH_PEER_DIR}")
    target_compile_definitions(${target} PRIVATE
      HYLE_BENCH_ABSL
      HYLE_BENCH_BOOST
      HYLE_BENCH_RIGTORP
      HYLE_BENCH_XXHASH)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_mark_absl_benchmark_consumer(${target})
    hyle_apply_warnings(${target})
  endfunction()

  add_executable(hyle_container_p1_bench p1_bench.cc)
  _hyle_configure_container_p1_bench(hyle_container_p1_bench)

  add_executable(hyle_container_p1_bench_placement p1_bench.cc)
  _hyle_configure_container_p1_bench(
    hyle_container_p1_bench_placement)
  if(MSVC)
    target_link_options(hyle_container_p1_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_container_p1_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "container P1 placement perturbation is unsupported")
  endif()

  hyle_register_benches(container
    hyle_container_p1_bench
    hyle_container_p1_bench_placement)
endif()
