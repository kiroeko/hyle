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

# Frozen P1 consumer/replay rows only. Keeping qualification in a dedicated
# executable avoids paying for legacy primitive/boundary diagnostics in every
# independent performance child.
add_executable(hyle_error_p1_bench p1_bench.cc)
target_link_libraries(hyle_error_p1_bench PRIVATE hyle_error nanobench)
hyle_apply_warnings(hyle_error_p1_bench)

if(HYLE_BENCH_ABSL)
  # Abseil is the mature-library rival for both equivalent contracts.
  target_compile_definitions(hyle_error_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_bench PRIVATE absl::status absl::statusor)
  hyle_mark_absl_benchmark_consumer(hyle_error_bench)
  target_compile_definitions(hyle_error_normal_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_normal_bench PRIVATE
    absl::status
    absl::status_builder
    absl::statusor)
  hyle_mark_absl_benchmark_consumer(hyle_error_normal_bench)
  target_compile_definitions(hyle_error_p1_bench PRIVATE HYLE_BENCH_ABSL)
  target_link_libraries(hyle_error_p1_bench PRIVATE
    absl::status
    absl::status_builder
    absl::statusor)
  hyle_mark_absl_benchmark_consumer(hyle_error_p1_bench)
endif()

set(HYLE_ERROR_BENCH_TARGETS
  hyle_error_bench
  hyle_error_normal_bench
  hyle_error_p1_bench)

# These short aggregates opt into one contract-equivalent code-placement
# perturbation. Both sides remain in the same translation unit and receive the
# same final-link/function-alignment treatment.
if(HYLE_STATIC)
  add_executable(hyle_error_bench_placement status_bench.cc)
  target_link_libraries(hyle_error_bench_placement PRIVATE
    hyle_error
    nanobench)
  hyle_apply_warnings(hyle_error_bench_placement)

  add_executable(hyle_error_normal_bench_placement normal_bench.cc)
  target_link_libraries(hyle_error_normal_bench_placement PRIVATE
    hyle_error
    nanobench)
  hyle_apply_warnings(hyle_error_normal_bench_placement)

  add_executable(hyle_error_p1_bench_placement p1_bench.cc)
  target_link_libraries(hyle_error_p1_bench_placement PRIVATE
    hyle_error
    nanobench)
  hyle_apply_warnings(hyle_error_p1_bench_placement)

  if(MSVC)
    target_link_options(hyle_error_bench_placement PRIVATE /FUNCTIONPADMIN:64)
    target_link_options(hyle_error_normal_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
    target_link_options(hyle_error_p1_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_error_bench_placement PRIVATE
      -falign-functions=64)
    target_compile_options(hyle_error_normal_bench_placement PRIVATE
      -falign-functions=64)
    target_compile_options(hyle_error_p1_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "error benchmark placement perturbation is unsupported by this compiler")
  endif()

  if(HYLE_BENCH_ABSL)
    target_compile_definitions(hyle_error_bench_placement PRIVATE
      HYLE_BENCH_ABSL)
    target_link_libraries(hyle_error_bench_placement PRIVATE
      absl::status
      absl::statusor)
    hyle_mark_absl_benchmark_consumer(hyle_error_bench_placement)

    target_compile_definitions(hyle_error_normal_bench_placement PRIVATE
      HYLE_BENCH_ABSL)
    target_link_libraries(hyle_error_normal_bench_placement PRIVATE
      absl::status
      absl::status_builder
      absl::statusor)
    hyle_mark_absl_benchmark_consumer(hyle_error_normal_bench_placement)

    target_compile_definitions(hyle_error_p1_bench_placement PRIVATE
      HYLE_BENCH_ABSL)
    target_link_libraries(hyle_error_p1_bench_placement PRIVATE
      absl::status
      absl::status_builder
      absl::statusor)
    hyle_mark_absl_benchmark_consumer(hyle_error_p1_bench_placement)
  endif()

  list(APPEND HYLE_ERROR_BENCH_TARGETS
    hyle_error_bench_placement
    hyle_error_normal_bench_placement
    hyle_error_p1_bench_placement)
endif()

hyle_register_benches(error
  ${HYLE_ERROR_BENCH_TARGETS})
