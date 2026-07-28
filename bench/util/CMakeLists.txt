# util-core gets a focused, normal-use certificate. It deliberately does not
# benchmark ScopeGuard/type_name (stack/constexpr mechanisms with no meaningful
# runtime peer) or the container-dependent Registry, which belongs util-runtime.
add_executable(hyle_util_core_bench util_core_bench.cc)
target_include_directories(hyle_util_core_bench PRIVATE "${PROJECT_SOURCE_DIR}/include")
target_compile_features(hyle_util_core_bench PRIVATE cxx_std_23)
target_link_libraries(hyle_util_core_bench PRIVATE nanobench)
hyle_apply_warnings(hyle_util_core_bench)

if(HYLE_BENCH_TBB)
  if(TARGET hyle_util_core_bench)
    target_compile_definitions(hyle_util_core_bench PRIVATE HYLE_BENCH_TBB)
    target_link_libraries(hyle_util_core_bench PRIVATE TBB::tbb)
  endif()
endif()

hyle_register_benches(util hyle_util_core_bench)
