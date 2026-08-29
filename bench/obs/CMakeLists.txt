# obs uses its own actual-workload certificate: gated formatting, structured
# records, fanout scaling, Status failure reporting and registry reads.
add_executable(hyle_obs_bench obs_bench.cc)
target_link_libraries(hyle_obs_bench PRIVATE hyle_obs nanobench)
hyle_apply_warnings(hyle_obs_bench)

set(_hyle_obs_benches hyle_obs_bench)
if(WIN32)
  # The ordinary nanobench process is pinned to one logical CPU by the shared
  # instrument.  Keep the W1-WIN concurrent contract separate so its eight
  # workers can each prove and retain a distinct physical-core placement.
  add_executable(hyle_obs_concurrent_bench concurrent_bench.cc)
  target_link_libraries(hyle_obs_concurrent_bench PRIVATE hyle_obs)
  hyle_apply_warnings(hyle_obs_concurrent_bench)
  list(APPEND _hyle_obs_benches hyle_obs_concurrent_bench)

  # The structured paired-process adapter compares two separately linked
  # products. The peer checkout is supplied only by a frozen performance
  # contract; it is not a product option or dependency. Build that checkout in
  # an isolated ExternalProject, then compile the exact same workload source
  # against its headers and static artifacts.
  if(DEFINED HYLE_BENCH_OBS_BASELINE_SOURCE)
    include(ExternalProject)
    cmake_path(ABSOLUTE_PATH HYLE_BENCH_OBS_BASELINE_SOURCE
      NORMALIZE OUTPUT_VARIABLE _hyle_obs_baseline_source)
    if(NOT EXISTS "${_hyle_obs_baseline_source}/CMakeLists.txt")
      message(FATAL_ERROR
        "HYLE_BENCH_OBS_BASELINE_SOURCE is not a HYLE source checkout")
    endif()
    set(_hyle_obs_baseline_build
      "${CMAKE_CURRENT_BINARY_DIR}/baseline-product")
    ExternalProject_Add(hyle_obs_baseline_product
      SOURCE_DIR "${_hyle_obs_baseline_source}"
      BINARY_DIR "${_hyle_obs_baseline_build}"
      CMAKE_GENERATOR "${CMAKE_GENERATOR}"
      CMAKE_GENERATOR_PLATFORM "${CMAKE_GENERATOR_PLATFORM}"
      CMAKE_ARGS
        -DHYLE_COMPONENTS=obs
        -DHYLE_STATIC=ON
        -DHYLE_BUILD_TESTS=OFF
        -DHYLE_BUILD_BENCH=OFF
        -DHYLE_INSTALL=OFF
        -DHYLE_SANITIZER=off
        -DHYLE_TRACE=OFF
        -DHYLE_TEST_INSTRUMENTATION=OFF
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
      BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR>
        --config Release --parallel 2 --target hyle_obs hyle_error
      INSTALL_COMMAND ""
      UPDATE_COMMAND ""
      BUILD_BYPRODUCTS
        "${_hyle_obs_baseline_build}/bin/Release/hyle_obs.lib"
        "${_hyle_obs_baseline_build}/bin/Release/hyle_error.lib")

    add_library(hyle_obs_baseline_archive STATIC IMPORTED GLOBAL)
    set_target_properties(hyle_obs_baseline_archive PROPERTIES
      IMPORTED_LOCATION
        "${_hyle_obs_baseline_build}/bin/Release/hyle_obs.lib")
    add_dependencies(hyle_obs_baseline_archive hyle_obs_baseline_product)
    add_library(hyle_error_baseline_archive STATIC IMPORTED GLOBAL)
    set_target_properties(hyle_error_baseline_archive PROPERTIES
      IMPORTED_LOCATION
        "${_hyle_obs_baseline_build}/bin/Release/hyle_error.lib")
    add_dependencies(hyle_error_baseline_archive hyle_obs_baseline_product)

    add_executable(hyle_obs_concurrent_baseline_bench concurrent_bench.cc)
    target_compile_features(hyle_obs_concurrent_baseline_bench PRIVATE cxx_std_23)
    target_compile_definitions(hyle_obs_concurrent_baseline_bench PRIVATE
      HYLE_STATIC)
    target_include_directories(hyle_obs_concurrent_baseline_bench PRIVATE
      "${_hyle_obs_baseline_source}/include")
    target_link_libraries(hyle_obs_concurrent_baseline_bench PRIVATE
      hyle_obs_baseline_archive hyle_error_baseline_archive)
    hyle_apply_warnings(hyle_obs_concurrent_baseline_bench)
    list(APPEND _hyle_obs_benches hyle_obs_concurrent_baseline_bench)
  endif()
endif()

# Tracy-upgrade evidence exists only in a trace-on build.  It compares HYLE's
# public macro seam with direct Tracy 0.14.1 macros under identical event
# contracts; direct backend use remains confined to this validation target.
if(TARGET hyle_tracy_client)
  add_executable(hyle_obs_tracy_bench tracy_bench.cc)
  target_link_libraries(hyle_obs_tracy_bench PRIVATE hyle_obs nanobench)
  hyle_apply_warnings(hyle_obs_tracy_bench)
  set_property(TARGET hyle_obs_tracy_bench
    PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

  if(HYLE_STATIC)
    add_executable(hyle_obs_tracy_bench_placement tracy_bench.cc)
    target_link_libraries(hyle_obs_tracy_bench_placement PRIVATE
      hyle_obs nanobench)
    if(MSVC)
      target_link_options(hyle_obs_tracy_bench_placement PRIVATE
        /FUNCTIONPADMIN:64)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(hyle_obs_tracy_bench_placement PRIVATE
        -falign-functions=64)
    else()
      message(FATAL_ERROR
        "obs trace placement perturbation is unsupported by this compiler")
    endif()
    hyle_apply_warnings(hyle_obs_tracy_bench_placement)
    set_property(TARGET hyle_obs_tracy_bench_placement
      PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  endif()
endif()

if(HYLE_BENCH_ABSL)
  if(TARGET hyle_obs_bench)
    target_compile_definitions(hyle_obs_bench PRIVATE HYLE_BENCH_ABSL)
    target_link_libraries(hyle_obs_bench PRIVATE absl::log absl::log_sink)
    hyle_mark_absl_benchmark_consumer(hyle_obs_bench)
  endif()
endif()

hyle_register_benches(obs ${_hyle_obs_benches})
if(TARGET hyle_obs_tracy_bench)
  hyle_register_benches(obs hyle_obs_tracy_bench)
endif()
if(TARGET hyle_obs_tracy_bench_placement)
  hyle_register_benches(obs hyle_obs_tracy_bench_placement)
endif()
