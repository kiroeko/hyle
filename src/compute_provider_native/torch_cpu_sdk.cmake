include_guard(GLOBAL)

# Complete CPU products require this SDK; exact factors derive private
# availability from their selected DAG. There is no public Provider option.
set(_HYLE_COMPUTE_HAVE_TORCH_CPU OFF)
if(_HYLE_COMPUTE_REQUEST_TORCH_CPU STREQUAL "OFF")
  return()
endif()
if(NOT (CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$" AND
        HYLE_TARGET_PROCESSOR STREQUAL "x86_64") AND
   NOT (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
        HYLE_TARGET_PROCESSOR STREQUAL "aarch64"))
  message(FATAL_ERROR "The TorchAO CPU adapter requires Windows/Linux x86_64 or macOS ARM64")
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
