# util-core gets a focused, normal-use certificate. It deliberately does not
# benchmark ScopeGuard/type_name (stack/constexpr mechanisms with no meaningful
# runtime peer) or the container-dependent Registry, which belongs util-runtime.
function(_hyle_configure_util_core_bench target)
  target_link_libraries(${target} PRIVATE hyle_util_core nanobench)
  set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
  hyle_apply_warnings(${target})

  if(HYLE_BENCH_ABSL)
    target_compile_definitions(${target} PRIVATE HYLE_BENCH_ABSL)
    target_link_libraries(${target} PRIVATE absl::function_ref)
    hyle_mark_absl_benchmark_consumer(${target})
  endif()

  if(HYLE_BENCH_TBB)
    target_compile_definitions(${target} PRIVATE HYLE_BENCH_TBB)
    target_link_libraries(${target} PRIVATE TBB::tbb)
  endif()

  if(HYLE_BENCH_BOOST)
    if(NOT EXISTS "${HYLE_BOOST_DIR}/boost/intrusive_ptr.hpp" AND
       NOT EXISTS
         "${HYLE_BOOST_DIR}/libs/smart_ptr/include/boost/intrusive_ptr.hpp")
      message(FATAL_ERROR
        "HYLE util_core Boost peer requires libs/smart_ptr; initialize that "
        "submodule in ${HYLE_BOOST_DIR}")
    endif()
    target_compile_definitions(${target} PRIVATE HYLE_BENCH_BOOST)
    target_link_libraries(${target} PRIVATE hyle_bench_boost_headers)
  endif()
endfunction()

add_executable(hyle_util_core_bench util_core_bench.cc)
_hyle_configure_util_core_bench(hyle_util_core_bench)

set(HYLE_UTIL_CORE_BENCH_TARGETS hyle_util_core_bench)
if(HYLE_STATIC AND
   (HYLE_BENCH_ABSL OR (HYLE_BENCH_TBB AND HYLE_BENCH_BOOST)))
  add_executable(hyle_util_core_bench_placement util_core_bench.cc)
  _hyle_configure_util_core_bench(hyle_util_core_bench_placement)
  if(MSVC)
    target_link_options(hyle_util_core_bench_placement PRIVATE
      /FUNCTIONPADMIN:64)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hyle_util_core_bench_placement PRIVATE
      -falign-functions=64)
  else()
    message(FATAL_ERROR
      "util_core benchmark placement perturbation is unsupported")
  endif()
  list(APPEND HYLE_UTIL_CORE_BENCH_TARGETS
    hyle_util_core_bench_placement)
endif()

hyle_register_benches(util_core ${HYLE_UTIL_CORE_BENCH_TARGETS})
