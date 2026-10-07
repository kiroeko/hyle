include_guard(GLOBAL)

hyle_auto_feature_option(HYLE_COMPUTE_TORCH_CPU
  "Use the original TorchAO/ATen CPU low precision programs" DEFAULT AUTO)
set(HYLE_COMPUTE_TORCH_CPU_ENABLED OFF)
if(HYLE_COMPUTE_TORCH_CPU STREQUAL "OFF")
  return()
endif()
if(NOT (CMAKE_SYSTEM_NAME MATCHES "^(Windows|Linux)$" AND
        HYLE_TARGET_PROCESSOR STREQUAL "x86_64") AND
   NOT (CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND
        HYLE_TARGET_PROCESSOR STREQUAL "aarch64"))
  if(HYLE_COMPUTE_TORCH_CPU STREQUAL "ON")
    message(FATAL_ERROR "The TorchAO CPU adapter requires Windows/Linux x86_64 or macOS ARM64")
  endif()
  return()
endif()

include("${CMAKE_CURRENT_LIST_DIR}/torch_cpu_fetch.cmake")
hyle_resolve_torch_cpu_sdk("${HYLE_SOURCE_DIR}/cmake/HyleDependencyLock.json" _hyle_torch_root)
set(HYLE_COMPUTE_TORCH_CPU_RESOLVED_ROOT "${_hyle_torch_root}" CACHE INTERNAL "Resolved locked CPU SDK" FORCE)
# An explicit root wins over package locations retained by an older build.
set(Torch_DIR "${_hyle_torch_root}/share/cmake/Torch")
set(Caffe2_DIR "${_hyle_torch_root}/share/cmake/Caffe2")
unset(TORCH_LIBRARY CACHE)
unset(c10_LIBRARY CACHE)
find_package(Torch 2.8.0 EXACT CONFIG REQUIRED
  PATHS "${_hyle_torch_root}/share/cmake/Torch" NO_DEFAULT_PATH)
hyle_register_torch_cpu_runtime("${_hyle_torch_root}")
include("${CMAKE_CURRENT_LIST_DIR}/HyleTorchCpuSdkConfig.cmake")
foreach(_hyle_torch_target IN ITEMS torch torch_cpu c10)
  if(TARGET ${_hyle_torch_target})
    get_target_property(_hyle_torch_global ${_hyle_torch_target} IMPORTED_GLOBAL)
    if(NOT _hyle_torch_global)
      set_property(TARGET ${_hyle_torch_target} PROPERTY IMPORTED_GLOBAL TRUE)
    endif()
  endif()
endforeach()
set(HYLE_COMPUTE_TORCH_CPU_ENABLED ON)
message(STATUS "HYLE: original TorchAO/ATen CPU low precision -- locked LibTorch 2.8.0")
