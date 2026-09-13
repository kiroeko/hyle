include_guard(GLOBAL)

# CUDA is a machine SDK, not a source dependency.  Declare and probe it only
# when the CUDA factor is part of the resolved component closure so a CPU-only
# source consumer never acquires a CUDA cache option or filesystem probe.
if(HYLE_BUILD_COMPUTE_BACKEND_CUDA)
  hyle_auto_feature_option(HYLE_COMPUTE_CUDA
    "Build the CUDA 13.3+ asynchronous Compute backend" DEFAULT OFF)
else()
  set(HYLE_COMPUTE_CUDA OFF)
endif()

function(hyle_configure_cuda_dependency)
  if(HYLE_COMPUTE_CUDA STREQUAL "OFF")
    message(STATUS "HYLE: CUDA backend OFF -- system SDK probe skipped")
    set(HYLE_COMPUTE_CUDA_ENABLED OFF PARENT_SCOPE)
    return()
  endif()

  if(HYLE_COMPUTE_CUDA STREQUAL "ON")
    find_package(CUDAToolkit 13.3 REQUIRED)
  else()
    find_package(CUDAToolkit 13.3 QUIET)
    if(NOT CUDAToolkit_FOUND)
      message(STATUS
        "HYLE: CUDA backend AUTO -- CUDA Toolkit 13.3+ not found; capability disabled")
      set(HYLE_COMPUTE_CUDA_ENABLED OFF PARENT_SCOPE)
      return()
    endif()
  endif()

  # A REQUIRED lookup already emits the authoritative configuration error.
  # Return here as well so CMake does not follow it with misleading secondary
  # diagnostics about imported targets that naturally were never created.
  if(NOT CUDAToolkit_FOUND)
    set(HYLE_COMPUTE_CUDA_ENABLED OFF PARENT_SCOPE)
    return()
  endif()

  foreach(_hyle_cuda_target IN ITEMS CUDA::cudart CUDA::cuda_driver)
    if(NOT TARGET ${_hyle_cuda_target})
      message(FATAL_ERROR
        "CUDA Toolkit ${CUDAToolkit_VERSION} did not define required target "
        "'${_hyle_cuda_target}'")
    endif()
  endforeach()
  unset(_hyle_cuda_target)

  # FindCUDAToolkit is intentionally called inside this function so its
  # implementation variables do not leak into the project namespace.  Export
  # the one public-header requirement under a HYLE-owned name for isolated
  # header validation; consumers receive the same path through CUDA::cudart.
  get_target_property(_hyle_cuda_include_dirs
    CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _hyle_cuda_include_dirs)
    message(FATAL_ERROR
      "CUDA::cudart did not publish the Toolkit include directory")
  endif()

  set(HYLE_COMPUTE_CUDA_ENABLED ON PARENT_SCOPE)
  set(HYLE_CUDA_PUBLIC_INCLUDE_DIRS
    "${_hyle_cuda_include_dirs}" PARENT_SCOPE)
  message(STATUS
    "HYLE: CUDA backend ${HYLE_COMPUTE_CUDA} -- found CUDA Toolkit "
    "${CUDAToolkit_VERSION} at '${CUDAToolkit_TARGET_DIR}'; capability enabled")
endfunction()
