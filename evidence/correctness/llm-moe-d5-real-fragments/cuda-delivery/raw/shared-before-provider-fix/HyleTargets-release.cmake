#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "hyle::_factor_error" for configuration "Release"
set_property(TARGET hyle::_factor_error APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_error PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_error.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_error.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_error )
list(APPEND _cmake_import_check_files_for_hyle::_factor_error "${_IMPORT_PREFIX}/lib/hyle_error.lib" "${_IMPORT_PREFIX}/bin/hyle_error.dll" )

# Import target "hyle::_factor_time" for configuration "Release"
set_property(TARGET hyle::_factor_time APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_time PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_time.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_time.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_time )
list(APPEND _cmake_import_check_files_for_hyle::_factor_time "${_IMPORT_PREFIX}/lib/hyle_time.lib" "${_IMPORT_PREFIX}/bin/hyle_time.dll" )

# Import target "hyle::_factor_platform" for configuration "Release"
set_property(TARGET hyle::_factor_platform APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_platform PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_platform.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_platform.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_platform )
list(APPEND _cmake_import_check_files_for_hyle::_factor_platform "${_IMPORT_PREFIX}/lib/hyle_platform.lib" "${_IMPORT_PREFIX}/bin/hyle_platform.dll" )

# Import target "hyle::_factor_platform_thread" for configuration "Release"
set_property(TARGET hyle::_factor_platform_thread APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_platform_thread PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_platform_thread.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_platform"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_platform_thread.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_platform_thread )
list(APPEND _cmake_import_check_files_for_hyle::_factor_platform_thread "${_IMPORT_PREFIX}/lib/hyle_platform_thread.lib" "${_IMPORT_PREFIX}/bin/hyle_platform_thread.dll" )

# Import target "hyle::_factor_mem" for configuration "Release"
set_property(TARGET hyle::_factor_mem APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_mem PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_mem.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_mem.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_mem )
list(APPEND _cmake_import_check_files_for_hyle::_factor_mem "${_IMPORT_PREFIX}/lib/hyle_mem.lib" "${_IMPORT_PREFIX}/bin/hyle_mem.dll" )

# Import target "hyle::_factor_simd" for configuration "Release"
set_property(TARGET hyle::_factor_simd APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_simd PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_simd.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_simd.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_simd )
list(APPEND _cmake_import_check_files_for_hyle::_factor_simd "${_IMPORT_PREFIX}/lib/hyle_simd.lib" "${_IMPORT_PREFIX}/bin/hyle_simd.dll" )

# Import target "hyle::_factor_container" for configuration "Release"
set_property(TARGET hyle::_factor_container APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_container PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_container.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_platform"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_container.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_container )
list(APPEND _cmake_import_check_files_for_hyle::_factor_container "${_IMPORT_PREFIX}/lib/hyle_container.lib" "${_IMPORT_PREFIX}/bin/hyle_container.dll" )

# Import target "hyle::_factor_compute" for configuration "Release"
set_property(TARGET hyle::_factor_compute APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_compute PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_compute.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_compute.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_compute )
list(APPEND _cmake_import_check_files_for_hyle::_factor_compute "${_IMPORT_PREFIX}/lib/hyle_compute.lib" "${_IMPORT_PREFIX}/bin/hyle_compute.dll" )

# Import target "hyle::_factor_graph" for configuration "Release"
set_property(TARGET hyle::_factor_graph APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_graph PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_graph.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_container"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_graph.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_graph )
list(APPEND _cmake_import_check_files_for_hyle::_factor_graph "${_IMPORT_PREFIX}/lib/hyle_graph.lib" "${_IMPORT_PREFIX}/bin/hyle_graph.dll" )

# Import target "hyle::_factor_compute_ops" for configuration "Release"
set_property(TARGET hyle::_factor_compute_ops APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_compute_ops PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_compute_ops.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_compute_ops.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_compute_ops )
list(APPEND _cmake_import_check_files_for_hyle::_factor_compute_ops "${_IMPORT_PREFIX}/lib/hyle_compute_ops.lib" "${_IMPORT_PREFIX}/bin/hyle_compute_ops.dll" )

# Import target "hyle::_factor_compute_backend_cuda" for configuration "Release"
set_property(TARGET hyle::_factor_compute_backend_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_compute_backend_cuda PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_compute_backend_cuda.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_platform_thread"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_compute_backend_cuda.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_compute_backend_cuda )
list(APPEND _cmake_import_check_files_for_hyle::_factor_compute_backend_cuda "${_IMPORT_PREFIX}/lib/hyle_compute_backend_cuda.lib" "${_IMPORT_PREFIX}/bin/hyle_compute_backend_cuda.dll" )

# Import target "hyle::_factor_compute_provider_cublas" for configuration "Release"
set_property(TARGET hyle::_factor_compute_provider_cublas APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_compute_provider_cublas PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_compute_provider_cublas.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_compute_backend_cuda;hyle::_factor_compute_ops;hyle::_factor_error;hyle::_factor_mem"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_compute_provider_cublas.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_compute_provider_cublas )
list(APPEND _cmake_import_check_files_for_hyle::_factor_compute_provider_cublas "${_IMPORT_PREFIX}/lib/hyle_compute_provider_cublas.lib" "${_IMPORT_PREFIX}/bin/hyle_compute_provider_cublas.dll" )

# Import target "hyle::_factor_compute_provider_cuda" for configuration "Release"
set_property(TARGET hyle::_factor_compute_provider_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_compute_provider_cuda PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_compute_provider_cuda.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_compute_backend_cuda;hyle::_factor_compute_ops;hyle::_factor_error;hyle::_factor_mem"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_compute_provider_cuda.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_compute_provider_cuda )
list(APPEND _cmake_import_check_files_for_hyle::_factor_compute_provider_cuda "${_IMPORT_PREFIX}/lib/hyle_compute_provider_cuda.lib" "${_IMPORT_PREFIX}/bin/hyle_compute_provider_cuda.dll" )

# Import target "hyle::_factor_compute_route_cuda" for configuration "Release"
set_property(TARGET hyle::_factor_compute_route_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_compute_route_cuda PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_compute_route_cuda.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_compute_backend_cuda;hyle::_factor_compute_provider_cublas;hyle::_factor_compute_provider_cuda;hyle::_factor_compute_provider_ggml_cuda"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_compute_route_cuda.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_compute_route_cuda )
list(APPEND _cmake_import_check_files_for_hyle::_factor_compute_route_cuda "${_IMPORT_PREFIX}/lib/hyle_compute_route_cuda.lib" "${_IMPORT_PREFIX}/bin/hyle_compute_route_cuda.dll" )

# Import target "hyle::_factor_graph_route_cuda" for configuration "Release"
set_property(TARGET hyle::_factor_graph_route_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_graph_route_cuda PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_graph_route_cuda.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_compute_backend_cuda;hyle::_factor_graph_provider_ggml_cuda"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_graph_route_cuda.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_graph_route_cuda )
list(APPEND _cmake_import_check_files_for_hyle::_factor_graph_route_cuda "${_IMPORT_PREFIX}/lib/hyle_graph_route_cuda.lib" "${_IMPORT_PREFIX}/bin/hyle_graph_route_cuda.dll" )

# Import target "hyle::_factor_compute_ggml_runtime" for configuration "Release"
set_property(TARGET hyle::_factor_compute_ggml_runtime APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_compute_ggml_runtime PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_compute_ggml_runtime.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_error;hyle::_factor_mem"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_compute_ggml_runtime.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_compute_ggml_runtime )
list(APPEND _cmake_import_check_files_for_hyle::_factor_compute_ggml_runtime "${_IMPORT_PREFIX}/lib/hyle_compute_ggml_runtime.lib" "${_IMPORT_PREFIX}/bin/hyle_compute_ggml_runtime.dll" )

# Import target "hyle::_factor_compute_provider_ggml_cuda" for configuration "Release"
set_property(TARGET hyle::_factor_compute_provider_ggml_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_compute_provider_ggml_cuda PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_compute_provider_ggml_cuda.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_compute_backend_cuda;hyle::_factor_compute_ops;hyle::_factor_compute_ggml_runtime;hyle::_factor_error;hyle::_factor_mem"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_compute_provider_ggml_cuda.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_compute_provider_ggml_cuda )
list(APPEND _cmake_import_check_files_for_hyle::_factor_compute_provider_ggml_cuda "${_IMPORT_PREFIX}/lib/hyle_compute_provider_ggml_cuda.lib" "${_IMPORT_PREFIX}/bin/hyle_compute_provider_ggml_cuda.dll" )

# Import target "hyle::_factor_graph_provider_ggml_cuda" for configuration "Release"
set_property(TARGET hyle::_factor_graph_provider_ggml_cuda APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(hyle::_factor_graph_provider_ggml_cuda PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/hyle_graph_provider_ggml_cuda.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "hyle::_factor_compute_provider_ggml_cuda;hyle::_factor_compute_ops;hyle::_factor_error;hyle::_factor_mem;hyle::_factor_graph"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/hyle_graph_provider_ggml_cuda.dll"
  )

list(APPEND _cmake_import_check_targets hyle::_factor_graph_provider_ggml_cuda )
list(APPEND _cmake_import_check_files_for_hyle::_factor_graph_provider_ggml_cuda "${_IMPORT_PREFIX}/lib/hyle_graph_provider_ggml_cuda.lib" "${_IMPORT_PREFIX}/bin/hyle_graph_provider_ggml_cuda.dll" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
