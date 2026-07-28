# bench: microbenchmarks (charter: nanobench). Built only with -DHYLE_BUILD_BENCH=ON.
# nanobench is a single-header lib (vendored third_party/nanobench); compile its
# impl into one static lib that bench exes link. hyle_add_bench(<mod>) (see
# HyleModule.cmake) makes hyle_<mod>_bench linked to hyle_<mod> + nanobench.
#
# Highway/xsimd are deliberately external and pinned here: benchmark rivals must
# remain unchanged after their product kernels were ported into HYLE. Product
# targets are independent of these external benchmark-only checkouts.
add_custom_target(hyle_benches)
set_target_properties(hyle_benches PROPERTIES FOLDER "hyle/validation")
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  string(TOUPPER "${_component}" _component_upper)
  if(HYLE_BUILD_${_component_upper})
    add_custom_target(hyle_${_component}_benches)
    set_target_properties(hyle_${_component}_benches PROPERTIES FOLDER "hyle/validation")
    add_dependencies(hyle_benches hyle_${_component}_benches)
  endif()
endforeach()
unset(_component)
unset(_component_upper)

add_library(nanobench STATIC nanobench_impl.cc)
set_target_properties(nanobench PROPERTIES FOLDER "hyle/third_party")
target_include_directories(nanobench SYSTEM PUBLIC
  "${PROJECT_SOURCE_DIR}/third_party/nanobench")
find_package(Threads REQUIRED)
target_link_libraries(nanobench PUBLIC Threads::Threads)

function(hyle_assert_bench_directory_registered directory)
  set(_expected_module "")
  if(ARGC GREATER 1)
    set(_expected_module "${ARGV1}")
  endif()
  get_property(_directory_targets DIRECTORY "${directory}"
    PROPERTY BUILDSYSTEM_TARGETS)
  foreach(_directory_target IN LISTS _directory_targets)
    get_target_property(_directory_target_type ${_directory_target} TYPE)
    if(_directory_target_type STREQUAL "EXECUTABLE")
      get_target_property(_bench_owner ${_directory_target} HYLE_BENCH_MODULE)
      if(NOT _bench_owner)
        message(FATAL_ERROR
          "Benchmark executable '${_directory_target}' in '${directory}' was not "
          "registered with a HYLE module aggregate")
      endif()
      if(_expected_module AND NOT _bench_owner STREQUAL _expected_module)
        message(FATAL_ERROR
          "Benchmark executable '${_directory_target}' in '${directory}' belongs to "
          "module '${_bench_owner}', expected '${_expected_module}'")
      endif()
    endif()
  endforeach()
endfunction()

# Shared benchmark peers are configured once; each module fragment decides
# whether and how its own targets consume them.
add_subdirectory(peers)

# Product modules own their benchmark registration beside benchmark sources.
# Each directory gets independent source/binary/variable scope.
foreach(_component IN LISTS HYLE_KNOWN_COMPONENTS)
  set(_hyle_bench_fragment "${CMAKE_CURRENT_SOURCE_DIR}/${_component}/CMakeLists.txt")
  if(TARGET hyle_${_component} AND EXISTS "${_hyle_bench_fragment}")
    add_subdirectory("${_component}")
    hyle_assert_bench_directory_registered(
      "${CMAKE_CURRENT_SOURCE_DIR}/${_component}" "${_component}")
  endif()
endforeach()
unset(_hyle_bench_fragment)
unset(_component)

# Explicit cross-module workloads do not masquerade as a product module.
add_subdirectory(integration)
hyle_assert_bench_directory_registered(
  "${CMAKE_CURRENT_SOURCE_DIR}/integration")
