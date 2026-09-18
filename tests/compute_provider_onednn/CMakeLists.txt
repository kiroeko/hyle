hyle_add_header_self_contained_test(compute_provider_onednn
  HEADER_DIR compute/providers
  HEADERS onednn.h)

if(HYLE_COMPUTE_ONEDNN_ENABLED AND TARGET hyle_compute_backend_cpu)
  hyle_add_test(compute_provider_onednn SOURCES onednn_provider_test.cc normalization_test.cc
    DEPS hyle_compute_backend_cpu hyle_simd)
  target_include_directories(hyle_compute_provider_onednn_test PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute_provider_onednn")
  if(TARGET hyle_compute_provider_native)
    hyle_add_test(compute_provider_onednn NAME composition
      SOURCES onednn_composition_test.cc
      DEPS hyle_compute_provider_native hyle_compute_backend_cpu)
  endif()
  # The executable must own oneDNN's object code so its global-new probe sees
  # vendor allocations. Shared-library builds retain the ordinary product
  # tests above; the static qualification shape is run explicitly once per
  # immutable provider candidate rather than on every ctest invocation.
  if(HYLE_STATIC)
    hyle_add_test(compute_provider_onednn NAME allocation
      SOURCES onednn_allocation_test.cc
      DEPS hyle_compute_backend_cpu
      NO_AUTORUN)
  endif()
  if(TARGET hyle_graph)
    hyle_add_test(compute_provider_onednn NAME graph
      SOURCES onednn_graph_test.cc
      DEPS hyle_graph hyle_compute_backend_cpu)
  endif()
endif()
