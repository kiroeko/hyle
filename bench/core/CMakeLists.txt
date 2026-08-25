# Core's allocation-free intrusive primitive: FIFO batch, refill + arbitrary
# cancel, read-only traversal, whole-batch handoff, and non-owning teardown.
# Boost uses the same member-hook auto-unlink contract. Whole-batch handoff is an
# O(1) mechanism diagnostic; std::list remains an owning/allocating floor only.
hyle_add_bench(core SOURCES core_bench.cc)

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
  if(EXISTS "${HYLE_BOOST_DIR}/boost/intrusive/list.hpp")
    target_include_directories(hyle_core_bench SYSTEM PRIVATE "${HYLE_BOOST_DIR}")
  elseif(EXISTS "${HYLE_BOOST_DIR}/libs/intrusive/include/boost/intrusive/list.hpp")
    foreach(_bl intrusive config assert core move static_assert)
      target_include_directories(hyle_core_bench SYSTEM PRIVATE
        "${HYLE_BOOST_DIR}/libs/${_bl}/include")
    endforeach()
  else()
    message(FATAL_ERROR "HYLE_BENCH_BOOST=ON but no boost/intrusive/list.hpp at ${HYLE_BOOST_DIR} (checked monolithic + modular libs/ layout; run `git submodule update --init` in the boost tree)")
  endif()
  target_compile_definitions(hyle_core_bench PRIVATE HYLE_BENCH_BOOST)
endif()
