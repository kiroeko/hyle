# HYLE CPU reference/optimized provider. Runtime and CPU backend stay separate
# so GPU-only maintenance builds do not acquire SIMD kernels.
include(HyleDispatch)
hyle_audit_in_tree_dependencies(ds4_quantization)
include("${CMAKE_CURRENT_SOURCE_DIR}/torch_cpu_sdk.cmake")

hyle_add_module(compute_provider_native SOURCES
  aten_cpu_provider.cc
  attention.cc
  cast.cc
  default_rules.cc
  ds4_quantization.cc
  elementwise.cc
  indexing.cc
  native_provider.cc
  normalization.cc
  reduction.cc
  transform.cc
  transformer.cc)
target_include_directories(hyle_compute_provider_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../compute"
  "${HYLE_SOURCE_DIR}/third_party/ds4_quantization")

if(HYLE_COMPUTE_TORCH_CPU_ENABLED)
  hyle_audit_in_tree_dependencies(torchao_cpu)
  target_sources(hyle_compute_provider_native PRIVATE torch_cpu_program.cc)
  target_include_directories(hyle_compute_provider_native PRIVATE
    "${HYLE_SOURCE_DIR}/third_party/torchao_cpu")
  target_include_directories(hyle_compute_provider_native SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
  if(TORCH_CXX_FLAGS)
    separate_arguments(_hyle_torch_flags NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
    target_compile_options(hyle_compute_provider_native PRIVATE ${_hyle_torch_flags})
  endif()
  # Use imported identities, not TORCH_LIBRARIES' absolute c10.lib path. A
  # relocated consumer resolves its own SDK through the dependency shim.
  target_link_libraries(hyle_compute_provider_native PRIVATE torch torch_cpu c10)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Let the consuming application's SDK RPATH reach this adapter's private
    # Torch dependency; a DT_RUNPATH here would suppress that ancestor search.
    # The adapter itself keeps HYLE's relocatable $ORIGIN installation path.
    target_link_options(hyle_compute_provider_native PRIVATE LINKER:--disable-new-dtags)
  endif()
  target_compile_definitions(hyle_compute_provider_native PRIVATE HYLE_COMPUTE_TORCH_CPU=1)
  set_property(TARGET hyle_compute_provider_native PROPERTY HYLE_TORCH_CPU_ENABLED TRUE)
  hyle_register_package_dependency(
    NAME torch_cpu_sdk
    OWNER compute_provider_native
    FEATURES compute_torch_cpu
    FIND_DEPENDENCIES HyleTorchCpuSdk
    DEVELOPMENT_FILE_SPECS
      "${CMAKE_CURRENT_SOURCE_DIR}/HyleTorchCpuSdkConfig.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCpuSdk|HyleTorchCpuSdkConfig.cmake"
      "${CMAKE_CURRENT_SOURCE_DIR}/torch_cpu_fetch.cmake|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCpuSdk|torch_cpu_fetch.cmake"
      "${HYLE_SOURCE_DIR}/cmake/HyleDependencyLock.json|${CMAKE_INSTALL_LIBDIR}/cmake/HyleTorchCpuSdk|HyleDependencyLock.json")
  hyle_register_package_dependency(
    NAME torchao_cpu
    OWNER compute_provider_native
    LOCKED_DEPENDENCIES torchao_cpu
    RUNTIME_FILE_SPECS
      "${HYLE_SOURCE_DIR}/third_party/torchao_cpu/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|TorchAO-CPU-LICENSE"
      "${HYLE_SOURCE_DIR}/third_party/torchao_cpu/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|TorchAO-CPU-NOTICE.md")
endif()

hyle_register_package_dependency(
  NAME ds4_quantization
  OWNER compute_provider_native
  LOCKED_DEPENDENCIES ds4_quantization
  RUNTIME_FILE_SPECS
    "${HYLE_SOURCE_DIR}/third_party/ds4_quantization/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ds4-quantization-LICENSE")

hyle_add_fat_kernel(
  TARGET hyle_compute_provider_native
  NAME compute_rope
  IMPL "src/compute/rope_kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
