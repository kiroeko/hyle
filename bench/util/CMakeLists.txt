hyle_add_bench(util SOURCES util_bench.cc)

if(HYLE_BENCH_ABSL)
  target_compile_definitions(hyle_util_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_util_bench PRIVATE absl::flat_hash_map)
  if(MSVC)
    # Abseil's MSVC prefetch fallback intentionally ignores its address when
    # the compiler has no matching intrinsic. Do not let that vendor-header
    # warning invalidate HYLE's own /WX benchmark build.
    target_compile_options(hyle_util_bench PRIVATE /wd4100 /wd4702)
  endif()
endif()
