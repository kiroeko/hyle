include_guard(GLOBAL)

# Complete CPU products require this SDK; exact factors derive private
# availability from their selected DAG. There is no public Provider option.
set(_HYLE_COMPUTE_HAVE_TORCH_CPU OFF)
if(DEFINED _HYLE_COMPUTE_REQUEST_CPU AND NOT _HYLE_COMPUTE_REQUEST_CPU)
  return()
endif()
include("${HYLE_SOURCE_DIR}/cmake/HyleComputeRoutes.cmake")
_hyle_compute_cpu_provider_supported(torch_cpu _hyle_torch_supported)
if(NOT _hyle_torch_supported)
  # Complete CPU products already fail route admission; exact maintenance
  # factors can still use their native implementation without this SDK.
  return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/torch_cpu_fetch.cmake")
hyle_resolve_torch_cpu_sdk("${HYLE_SOURCE_DIR}/cmake/HyleDependencyLock.json" _hyle_torch_root)
_hyle_prepare_torch_cpu_sdk("${_hyle_torch_root}")
find_package(Torch 2.8.0 EXACT CONFIG REQUIRED
  PATHS "${_hyle_torch_root}/share/cmake/Torch" NO_DEFAULT_PATH)
include("${CMAKE_CURRENT_LIST_DIR}/HyleTorchCpuSdkConfig.cmake")
_hyle_promote_torch_cpu_targets()
set(_HYLE_COMPUTE_HAVE_TORCH_CPU ON)
message(STATUS "HYLE: original TorchAO/ATen CPU low precision -- locked LibTorch 2.8.0")
