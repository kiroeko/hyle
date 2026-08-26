# Allocation-free intrusive primitive: FIFO batch, refill + arbitrary
# cancel, read-only traversal, whole-batch handoff, non-owning teardown, and
# auto-unlink node retirement. Boost uses matching member-hook auto_unlink and
# safe_link contracts with constant-time size disabled. Whole-batch handoff is
# an O(1) mechanism diagnostic; std::list remains an owning/allocating floor.
hyle_add_bench(container_intrusive
  SOURCES intrusive_list_bench.cc storage_observer.cc)
set_property(TARGET hyle_container_intrusive_bench
  PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

# Qualification controls stay in a separate executable so A/A and known-delta
# probes cannot perturb the product comparison's function placement.
add_executable(hyle_container_intrusive_instrument_bench
  instrument_bench.cc storage_observer.cc)
set_property(TARGET hyle_container_intrusive_instrument_bench
  PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
target_link_libraries(hyle_container_intrusive_instrument_bench PRIVATE
  hyle_container_intrusive
  nanobench)
hyle_apply_warnings(hyle_container_intrusive_instrument_bench)
hyle_register_bench(container_intrusive
  hyle_container_intrusive_instrument_bench)

# boost::intrusive::list -- the canonical intrusive-container SOTA (header-only).
# Enable with -DHYLE_BENCH_BOOST=ON; point HYLE_BOOST_DIR at a boost tree (the
# superproject root with boost/intrusive/list.hpp reachable, e.g. a full boost
# clone or system boost). Header-only -> just an include path, no link/build.
option(HYLE_BENCH_BOOST "build IntrusiveList bench vs boost::intrusive::list" OFF)
if(HYLE_BENCH_BOOST)
  set(HYLE_BOOST_DIR "${PROJECT_SOURCE_DIR}/../boost" CACHE PATH "boost tree root")
  # Accept BOTH boost layouts: monolithic (boost/intrusive/list.hpp at root, e.g.
  # system boost) OR modular superproject (libs/<lib>/include/boost/...; a fresh
  # boostorg clone with submodules checked out). boost::intrusive is header-only but
  # pulls a small dep closure (config/assert/core/move/static_assert) -- add each
  # lib's include/ dir in the modular case.
  if(NOT EXISTS "${HYLE_BOOST_DIR}/boost/intrusive/list.hpp" AND
     NOT EXISTS "${HYLE_BOOST_DIR}/libs/intrusive/include/boost/intrusive/list.hpp")
    message(FATAL_ERROR "HYLE_BENCH_BOOST=ON but no boost/intrusive/list.hpp at ${HYLE_BOOST_DIR} (checked monolithic + modular libs/ layout; run `git submodule update --init` in the boost tree)")
  endif()

  function(_hyle_configure_container_intrusive_peer target)
    if(EXISTS "${HYLE_BOOST_DIR}/boost/intrusive/list.hpp")
      target_include_directories(${target} SYSTEM PRIVATE "${HYLE_BOOST_DIR}")
    else()
      foreach(_bl intrusive config assert core move static_assert)
        target_include_directories(${target} SYSTEM PRIVATE
          "${HYLE_BOOST_DIR}/libs/${_bl}/include")
      endforeach()
    endif()
    target_compile_definitions(${target} PRIVATE HYLE_BENCH_BOOST)
  endfunction()

  _hyle_configure_container_intrusive_peer(hyle_container_intrusive_bench)

  # Short pointer-relink kernels need a second, contract-identical placement.
  # MSVC perturbs final-link padding without changing compiler objects. GNU and
  # AppleClang apply the same function-alignment option to the whole comparison
  # TU, so HYLE and Boost remain symmetric in one executable.
  if(HYLE_STATIC)
    add_executable(hyle_container_intrusive_bench_placement
      intrusive_list_bench.cc storage_observer.cc)
    target_link_libraries(hyle_container_intrusive_bench_placement PRIVATE
      hyle_container_intrusive
      nanobench)
    if(MSVC)
      target_link_options(hyle_container_intrusive_bench_placement PRIVATE
        /FUNCTIONPADMIN:64)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
      target_compile_options(hyle_container_intrusive_bench_placement PRIVATE
        -falign-functions=64)
    else()
      message(FATAL_ERROR
        "container_intrusive placement perturbation is unsupported by this compiler")
    endif()
    hyle_apply_warnings(hyle_container_intrusive_bench_placement)
    set_property(TARGET hyle_container_intrusive_bench_placement
      PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    _hyle_configure_container_intrusive_peer(
      hyle_container_intrusive_bench_placement)
    hyle_register_bench(container_intrusive
      hyle_container_intrusive_bench_placement)
  endif()
endif()
