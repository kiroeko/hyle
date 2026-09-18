# The ordinary CPU route now owns the complete-region comparison.

# The normalization adoption experiment compares complete Direct/Graph paths.
# Explicit provider selection is confined to this development benchmark.
if(HYLE_STATIC AND HYLE_COMPUTE_ONEDNN_ENABLED
    AND TARGET hyle_compute_provider_native)
  foreach(_variant IN ITEMS primary placement)
    set(_target hyle_onednn_normalization_bench)
    if(_variant STREQUAL "placement")
      string(APPEND _target _placement)
      add_executable(${_target} normalization_placement.cc normalization_p1_bench.cc)
      target_compile_definitions(${_target} PRIVATE HYLE_NORM_PLACEMENT=1)
    else()
      add_executable(${_target} normalization_p1_bench.cc)
      target_compile_definitions(${_target} PRIVATE HYLE_NORM_PLACEMENT=0)
    endif()
    target_link_libraries(${_target} PRIVATE
      hyle_graph_provider_onednn hyle_compute_provider_onednn
      hyle_compute_provider_native hyle_compute_backend_cpu nanobench)
    set_property(TARGET ${_target} PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)
    if(MSVC)
      target_link_options(${_target} PRIVATE
        "/MAP:${CMAKE_CURRENT_BINARY_DIR}/${_target}.map")
    endif()
    hyle_apply_warnings(${_target})
    hyle_register_bench(graph_provider_onednn ${_target})
  endforeach()
endif()
