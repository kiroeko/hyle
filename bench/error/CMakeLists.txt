# Information-free Status/Result paths: compare equivalent code-only carriers
# and success propagation. This executable intentionally excludes diagnostics.
add_executable(hyle_error_bench status_bench.cc)
target_link_libraries(hyle_error_bench PRIVATE hyle_error nanobench)
hyle_apply_warnings(hyle_error_bench)

# Information-bearing user workflows live in a second executable: messages,
# semantic context, native detail, errno translation, and formatting.
add_executable(hyle_error_normal_bench normal_bench.cc)
target_link_libraries(hyle_error_normal_bench PRIVATE hyle_error nanobench)
hyle_apply_warnings(hyle_error_normal_bench)

if(HYLE_BENCH_ABSL)
  # Abseil is the mature-library rival for both equivalent contracts.
  target_compile_definitions(hyle_error_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_bench PRIVATE absl::status absl::statusor)
  target_compile_definitions(hyle_error_normal_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_normal_bench PRIVATE
    absl::status
    absl::status_builder
    absl::statusor)
endif()

hyle_register_benches(error
  hyle_error_bench
  hyle_error_normal_bench)
