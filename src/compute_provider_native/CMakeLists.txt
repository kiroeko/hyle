# HYLE CPU reference/optimized provider. Runtime and CPU backend stay separate
# so GPU-only maintenance builds do not acquire SIMD kernels.
include(HyleDispatch)

hyle_add_module(compute_provider_native SOURCES native_provider.cc transform.cc)
target_include_directories(hyle_compute_provider_native PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../compute")

hyle_add_fat_kernel(
  TARGET hyle_compute_provider_native
  NAME compute_matmul
  IMPL "src/compute/matmul_kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
