hyle_add_bench(util_runtime SOURCES util_bench.cc)

if(HYLE_BENCH_ABSL)
  target_compile_definitions(hyle_util_runtime_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_util_runtime_bench PRIVATE absl::flat_hash_map)
  # The peer is intentionally outside HYLE's warning contract. Keep its
  # template headers visible to the benchmark without turning Abseil's own
  # pedantic extensions into HYLE_WERROR failures on GCC/Clang.
  target_include_directories(hyle_util_runtime_bench SYSTEM PRIVATE
    "${HYLE_ABSL_DIR}")
  hyle_mark_absl_benchmark_consumer(hyle_util_runtime_bench)
  if(MSVC)
    # Abseil's MSVC prefetch fallback intentionally ignores its address when
    # the compiler has no matching intrinsic. Do not let that vendor-header
    # warning invalidate HYLE's own /WX benchmark build.
    target_compile_options(hyle_util_runtime_bench PRIVATE /wd4100)
  endif()
endif()
