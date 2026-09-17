if(HYLE_STATIC AND HYLE_COMPUTE_VULKAN_ENABLED)
  file(TO_CMAKE_PATH "${HYLE_VULKAN_REFERENCE_SOURCE_DIR}" _reference_source)
  if(NOT EXISTS "${_reference_source}/src/ggml-vulkan/ggml-vulkan.cpp")
    message(FATAL_ERROR "Vulkan benchmarks require the pinned independent HYLE_VULKAN_REFERENCE_SOURCE_DIR")
  endif()
  set(_reference_output "${CMAKE_CURRENT_BINARY_DIR}/reference-shaders")
  file(GLOB _reference_inputs CONFIGURE_DEPENDS
    "${_reference_source}/src/ggml-vulkan/vulkan-shaders/*")
  add_custom_command(OUTPUT "${_reference_output}/reference_spirv.h"
    COMMAND "${HYLE_VULKAN_PYTHON}" "${CMAKE_CURRENT_SOURCE_DIR}/reference_shaders.py"
      --peer "${_reference_source}" --compiler "${HYLE_VULKAN_GLSLANG}"
      --output "${_reference_output}"
    DEPENDS reference_shaders.py ${_reference_inputs} VERBATIM)
  add_library(hyle_vulkan_reference STATIC reference.cc "${_reference_output}/reference_spirv.h")
  target_include_directories(hyle_vulkan_reference PRIVATE
    "${_reference_output}" "${HYLE_SOURCE_DIR}/src/compute_route_vulkan")
  target_link_libraries(hyle_vulkan_reference PRIVATE hyle_compute_route_vulkan)
  hyle_apply_warnings(hyle_vulkan_reference)
  set_property(TARGET hyle_vulkan_reference PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

  function(hyle_configure_vulkan_ffn_bench target owner graph_mode placement natural)
    target_compile_definitions(${target} PRIVATE HYLE_VULKAN_GRAPH_BENCH=${graph_mode}
      HYLE_VULKAN_PLACEMENT=${placement} HYLE_VULKAN_NATIVE_COST=${natural})
    if(MSVC)
      target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS)
      target_link_options(${target} PRIVATE "/MAP:${CMAKE_CURRENT_BINARY_DIR}/${target}.map")
    elseif(APPLE)
      target_link_options(${target} PRIVATE "-Wl,-map,${CMAKE_CURRENT_BINARY_DIR}/${target}.map")
    endif()
    target_link_libraries(${target} PRIVATE hyle_${owner} hyle_vulkan_reference nanobench)
    set_property(TARGET ${target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    hyle_apply_warnings(${target})
  endfunction()
  add_executable(hyle_compute_route_vulkan_bench route_p1_bench.cc)
  hyle_configure_vulkan_ffn_bench(hyle_compute_route_vulkan_bench compute_route_vulkan 0 0 0)
  add_executable(hyle_compute_route_vulkan_bench_placement placement.cc route_p1_bench.cc)
  hyle_configure_vulkan_ffn_bench(hyle_compute_route_vulkan_bench_placement compute_route_vulkan 0 1 0)
  add_executable(hyle_compute_vulkan_native_cost_bench route_p1_bench.cc)
  hyle_configure_vulkan_ffn_bench(hyle_compute_vulkan_native_cost_bench compute_route_vulkan 0 0 1)
  hyle_register_benches(compute_route_vulkan hyle_compute_route_vulkan_bench
    hyle_compute_route_vulkan_bench_placement hyle_compute_vulkan_native_cost_bench)
endif()
