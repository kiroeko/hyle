# Benchmark-only peers shared by more than one product module. Keep only
# dependency configuration here; target wiring belongs to each module fragment.

# Optional oneTBB rivals. Declare the switch only when at least one selected
# benchmark owner can consume it; a Core-only benchmark cache must not advertise
# unrelated peer libraries.
set(_hyle_bench_tbb_owner FALSE)
foreach(_hyle_owner IN ITEMS platform mem container util_core)
  string(TOUPPER "${_hyle_owner}" _hyle_owner_upper)
  if(HYLE_BUILD_${_hyle_owner_upper})
    set(_hyle_bench_tbb_owner TRUE)
  endif()
endforeach()
if(_hyle_bench_tbb_owner)
  option(HYLE_BENCH_TBB "build relevant benchmarks vs oneTBB" OFF)
else()
  set(HYLE_BENCH_TBB OFF)
endif()
if(HYLE_BENCH_TBB)
  set(HYLE_TBB_DIR "${HYLE_SOURCE_DIR}/../oneTBB" CACHE PATH
      "local oneTBB source snapshot")
  if(NOT EXISTS "${HYLE_TBB_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_TBB=ON but no oneTBB at ${HYLE_TBB_DIR}")
  endif()
  # Pass benchmark-only policy as directory variables. With CMP0077's modern
  # behavior, the peer's option() calls honor these without forcing unrelated
  # entries into HYLE's cache.
  set(TBB_TEST OFF)
  set(TBBMALLOC_BUILD ON)
  set(TBB_EXAMPLES OFF)
  # HYLE performance contracts disable LTO for both sides. oneTBB owns a
  # separate IPO switch and otherwise enables it even when the parent cache
  # sets CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF.
  set(TBB_ENABLE_IPO OFF)
  # Recent oneTBB defaults the optional Thread Composability Manager ON. Its
  # second configure then treats that cached default as an explicit request and
  # hard-requires HWLOC. Benchmarks only need normal oneTBB targets.
  set(TCM_BUILD OFF)
  set(TCM_TEST OFF)
  add_subdirectory(
    "${HYLE_TBB_DIR}"
    "${HYLE_BINARY_DIR}/bench/peers/onetbb"
    EXCLUDE_FROM_ALL SYSTEM)

  # oneTBB's Windows targets are DLLs even when the HYLE candidate is static.
  # Contract runners launch build-tree executables from bin/<config>, so stage
  # the exact linked peer runtimes beside each consumer instead of relying on a
  # developer's PATH.
  function(hyle_stage_tbb_benchmark_runtime target)
    if(NOT TARGET ${target})
      message(FATAL_ERROR
        "hyle_stage_tbb_benchmark_runtime: unknown target '${target}'")
    endif()
    if(WIN32)
      add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:TBB::tbb>"
                "$<TARGET_FILE:TBB::tbbmalloc>"
                "$<TARGET_FILE_DIR:${target}>"
        COMMAND_EXPAND_LISTS
        VERBATIM)
    endif()
  endfunction()
endif()

# Optional Abseil rivals are likewise owner-scoped. They cover more than the
# original SwissTable comparison today, but remain benchmark-only dependencies.
set(_hyle_bench_absl_owner FALSE)
foreach(_hyle_owner IN ITEMS container error obs util)
  string(TOUPPER "${_hyle_owner}" _hyle_owner_upper)
  if(HYLE_BUILD_${_hyle_owner_upper})
    set(_hyle_bench_absl_owner TRUE)
  endif()
endforeach()
if(_hyle_bench_absl_owner)
  option(HYLE_BENCH_ABSL "build relevant benchmarks vs Abseil" OFF)
else()
  set(HYLE_BENCH_ABSL OFF)
endif()
if(HYLE_BENCH_ABSL)
  set(HYLE_ABSL_DIR "${HYLE_SOURCE_DIR}/../abseil-cpp" CACHE PATH "cloned abseil-cpp")
  if(NOT EXISTS "${HYLE_ABSL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "HYLE_BENCH_ABSL=ON but no abseil at ${HYLE_ABSL_DIR}")
  endif()
  set(ABSL_PROPAGATE_CXX_STD ON)
  set(ABSL_ENABLE_INSTALL OFF)
  add_subdirectory(
    "${HYLE_ABSL_DIR}"
    "${HYLE_BINARY_DIR}/bench/peers/abseil"
    EXCLUDE_FROM_ALL SYSTEM)

  # MSVC attributes warnings from instantiated external templates to the
  # consuming translation unit. Current Abseil has constexpr fallback branches
  # that intentionally diagnose as C4702 even with /external:W0. Apply this
  # exception only to benchmark targets that explicitly consume Abseil.
  function(hyle_mark_absl_benchmark_consumer target)
    if(NOT TARGET ${target})
      message(FATAL_ERROR "Unknown Abseil benchmark consumer: ${target}")
    endif()
    if(MSVC)
      target_compile_options(${target} PRIVATE /wd4702)
    endif()
  endfunction()
endif()

unset(_hyle_bench_tbb_owner)
unset(_hyle_bench_absl_owner)
unset(_hyle_owner)
unset(_hyle_owner_upper)

# Optional Boost header-only rivals. Keep source discovery and the modular
# superproject include closure in one place: individual benchmark directories
# only decide which targets instantiate which peer capability.
set(_hyle_bench_boost_owner FALSE)
foreach(_hyle_owner IN ITEMS container_intrusive platform util_core mem)
  string(TOUPPER "${_hyle_owner}" _hyle_owner_upper)
  if(HYLE_BUILD_${_hyle_owner_upper})
    set(_hyle_bench_boost_owner TRUE)
  endif()
endforeach()
if(_hyle_bench_boost_owner)
  option(HYLE_BENCH_BOOST "build relevant benchmarks vs Boost" OFF)
else()
  set(HYLE_BENCH_BOOST OFF)
endif()
if(HYLE_BENCH_BOOST)
  set(HYLE_BOOST_DIR "${HYLE_SOURCE_DIR}/../boost" CACHE PATH
      "local Boost source snapshot")
  if(NOT EXISTS "${HYLE_BOOST_DIR}/boost/config.hpp" AND
     NOT EXISTS "${HYLE_BOOST_DIR}/libs/config/include/boost/config.hpp")
    message(FATAL_ERROR
      "HYLE_BENCH_BOOST=ON but no monolithic or modular Boost tree exists at "
      "${HYLE_BOOST_DIR}")
  endif()

  add_library(hyle_bench_boost_headers INTERFACE)
  if(EXISTS "${HYLE_BOOST_DIR}/boost/config.hpp")
    target_include_directories(hyle_bench_boost_headers SYSTEM INTERFACE
      "${HYLE_BOOST_DIR}")
  else()
    foreach(_hyle_boost_library IN ITEMS
        assert config container core interprocess intrusive move predef smart_ptr
        static_assert winapi)
      if(EXISTS
          "${HYLE_BOOST_DIR}/libs/${_hyle_boost_library}/include")
        target_include_directories(hyle_bench_boost_headers SYSTEM INTERFACE
          "${HYLE_BOOST_DIR}/libs/${_hyle_boost_library}/include")
      endif()
    endforeach()
    unset(_hyle_boost_library)
  endif()
endif()

unset(_hyle_bench_boost_owner)
unset(_hyle_owner)
unset(_hyle_owner_upper)

# Marl supplies the bounded object-pool peer for mem and the scheduler/fiber
# peers for sched. Discover the checkout once; module fragments own the exact
# source adapter/target they need.
set(_hyle_bench_marl_owner FALSE)
foreach(_hyle_owner IN ITEMS mem sched)
  string(TOUPPER "${_hyle_owner}" _hyle_owner_upper)
  if(HYLE_BUILD_${_hyle_owner_upper})
    set(_hyle_bench_marl_owner TRUE)
  endif()
endforeach()
if(_hyle_bench_marl_owner)
  option(HYLE_BENCH_MARL "build relevant benchmarks vs Marl" OFF)
else()
  set(HYLE_BENCH_MARL OFF)
endif()
if(HYLE_BENCH_MARL)
  set(HYLE_MARL_DIR "${HYLE_SOURCE_DIR}/../marl" CACHE PATH
      "local Marl source snapshot")
  if(NOT EXISTS "${HYLE_MARL_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "HYLE_BENCH_MARL=ON but no Marl checkout at ${HYLE_MARL_DIR}")
  endif()
endif()

unset(_hyle_bench_marl_owner)
unset(_hyle_owner)
unset(_hyle_owner_upper)
