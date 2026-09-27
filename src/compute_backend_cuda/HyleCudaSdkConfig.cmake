# Private package shared by source and installed CUDA products. CMake's
# FindCUDAToolkit describes Windows import libraries as UNKNOWN targets, so
# TARGET_RUNTIME_DLLS cannot otherwise see their actual runtime files.
include(CMakeFindDependencyMacro)
find_dependency(CUDAToolkit 13.3)

if(WIN32)
  function(_hyle_cuda_sdk_runtime library)
    set(_target "HyleCudaSdk::${library}_runtime")
    if(TARGET "${_target}")
      return()
    endif()
    get_target_property(_type "CUDA::${library}" TYPE)
    get_target_property(_import "CUDA::${library}" IMPORTED_LOCATION)
    if(_type STREQUAL "SHARED_LIBRARY" AND _import MATCHES "\\.[dD][lL][lL]$")
      return()
    endif()
    if(NOT EXISTS "${_import}" OR NOT _import MATCHES "\\.[lL][iI][bB]$")
      message(FATAL_ERROR "CUDA::${library} must identify its Windows SDK import library")
    endif()
    find_file(_runtime
      NAMES "${library}64_${CUDAToolkit_VERSION_MAJOR}.dll"
      PATHS "${CUDAToolkit_BIN_DIR}/x64" "${CUDAToolkit_BIN_DIR}"
            "${CUDAToolkit_LIBRARY_ROOT}/bin/x64" "${CUDAToolkit_LIBRARY_ROOT}/bin"
      NO_DEFAULT_PATH NO_CACHE REQUIRED)
    add_library("${_target}" SHARED IMPORTED GLOBAL)
    set_target_properties("${_target}" PROPERTIES
      IMPORTED_IMPLIB "${_import}" IMPORTED_LOCATION "${_runtime}")
    # Preserve the SDK target's original includes, dependencies and selected
    # import library. This leaf adds its loader identity to the same link graph.
    target_link_libraries("CUDA::${library}" INTERFACE "${_target}")
  endfunction()
  foreach(_library IN ITEMS cudart cublasLt cublas)
    _hyle_cuda_sdk_runtime("${_library}")
  endforeach()
  if(TARGET HyleCudaSdk::cublas_runtime AND TARGET HyleCudaSdk::cublasLt_runtime)
    # cuBLAS itself imports cuBLASLt, including consumers that do not call the
    # latter directly. No system driver DLL is copied or linked here.
    set_property(TARGET HyleCudaSdk::cublas_runtime APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES HyleCudaSdk::cublasLt_runtime)
  endif()
endif()
