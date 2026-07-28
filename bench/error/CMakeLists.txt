# error-module bench: Status/Result (zero-dependency root, every hot-path return type)
# vs std::expected + absl::Status/StatusOr (the same tagged-pointer design; gated
# -DHYLE_BENCH_ABSL, absl::status/statusor linked in the absl block below). Own exe.
add_executable(hyle_error_bench status_bench.cc)
target_link_libraries(hyle_error_bench PRIVATE hyle_error nanobench)
hyle_apply_warnings(hyle_error_bench)

# Real error workflows live in a second TU/executable so extending them cannot
# perturb the core carrier microbench's whole-TU inlining/code layout.
add_executable(hyle_error_normal_bench normal_bench.cc)
target_link_libraries(hyle_error_normal_bench PRIVATE hyle_error nanobench)
hyle_apply_warnings(hyle_error_normal_bench)

if(HYLE_BENCH_ABSL)
  # absl::Status/StatusOr are the same tagged-pointer design as HYLE Status -> the
  # error bench's SOTA rival (same gate). absl::statusor pulls in absl::status.
  target_compile_definitions(hyle_error_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_bench PRIVATE absl::status absl::status_builder absl::statusor)
  target_compile_definitions(hyle_error_normal_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_normal_bench PRIVATE absl::status)
endif()

hyle_register_benches(error
  hyle_error_bench
  hyle_error_normal_bench)
