# HYLE CPU reference/optimized provider. Runtime and CPU backend stay separate
# so GPU-only maintenance builds do not acquire SIMD kernels.
include(HyleDispatch)
hyle_audit_in_tree_dependencies(ds4_quantization)

hyle_add_module(compute_provider_native SOURCES
  attention.cc
  cast.cc
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

hyle_register_package_dependency(
  NAME ds4_quantization
  OWNER compute_provider_native
  LOCKED_DEPENDENCIES ds4_quantization
  RUNTIME_FILE_SPECS
    "${HYLE_SOURCE_DIR}/third_party/ds4_quantization/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|ds4-quantization-LICENSE")

hyle_add_fat_kernel(
  TARGET hyle_compute_provider_native
  NAME compute_matmul
  IMPL "src/compute/matmul_kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)

hyle_add_fat_kernel(
  TARGET hyle_compute_provider_native
  NAME compute_rope
  IMPL "src/compute/rope_kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
