# CPU-only machine SDK, private to the Native adapter. This same check runs in
# a static installed consumer; finding a GPU Torch package is not CPU isolation.
include(CMakeFindDependencyMacro)
find_dependency(Torch 2.8.0 EXACT CONFIG)
if(NOT EXISTS "${TORCH_INSTALL_PREFIX}/build-version")
  message(FATAL_ERROR "HYLE low precision requires a versioned LibTorch CPU SDK")
endif()
file(READ "${TORCH_INSTALL_PREFIX}/build-version" _hyle_torch_version)
string(STRIP "${_hyle_torch_version}" _hyle_torch_version)
if(NOT _hyle_torch_version STREQUAL "2.8.0+cpu" OR CAFFE2_USE_CUDA)
  message(FATAL_ERROR "HYLE requires CPU-only LibTorch 2.8.0+cpu, found '${_hyle_torch_version}'")
endif()
set(HyleTorchCpuSdk_FOUND TRUE)
unset(_hyle_torch_version)
