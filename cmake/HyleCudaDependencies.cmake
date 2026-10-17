include_guard(GLOBAL)

# CUDA is a machine SDK, not a source dependency.  Declare and probe it only
# when the CUDA factor is part of the resolved component closure so a CPU-only
# source consumer never acquires a CUDA cache option or filesystem probe.
# Private availability is derived from the selected DAG by HyleComputeRoutes.

function(hyle_configure_cuda_dependency)
  if(_HYLE_COMPUTE_REQUEST_CUDA STREQUAL "OFF")
    message(STATUS "HYLE: CUDA backend OFF -- system SDK probe skipped")
    set(_HYLE_COMPUTE_HAVE_CUDA OFF PARENT_SCOPE)
    return()
  endif()

  find_package(CUDAToolkit 13.3 REQUIRED)

  # A REQUIRED lookup already emits the authoritative configuration error.
  # Return here as well so CMake does not follow it with misleading secondary
  # diagnostics about imported targets that naturally were never created.
  if(NOT CUDAToolkit_FOUND)
    set(_HYLE_COMPUTE_HAVE_CUDA OFF PARENT_SCOPE)
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

  # Establish the route default before sibling native projects are entered.
  # A default local to the backend cannot reach ggml's CUDA build. Explicit
  # CMake/CUDAARCHS selections retain their normal precedence.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if("$ENV{CUDAARCHS}" STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES 80 89-real PARENT_SCOPE)
    else()
      set(CMAKE_CUDA_ARCHITECTURES "$ENV{CUDAARCHS}" PARENT_SCOPE)
    endif()
  endif()

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

  set(_HYLE_COMPUTE_HAVE_CUDA ON PARENT_SCOPE)
  set(HYLE_CUDA_PUBLIC_INCLUDE_DIRS
    "${_hyle_cuda_include_dirs}" PARENT_SCOPE)
  message(STATUS
    "HYLE: CUDA backend ${_HYLE_COMPUTE_REQUEST_CUDA} -- found CUDA Toolkit "
    "${CUDAToolkit_VERSION} at '${CUDAToolkit_TARGET_DIR}'; capability enabled")
endfunction()
