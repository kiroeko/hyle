# Compute owns the backend-neutral device/context/tensor and prepared-operation
# contracts. The current product sources provide the synchronous CPU route;
# optional providers are added as separate implementation factors.
include(HyleDispatch)

hyle_add_module(compute SOURCES device.cc provider.cc tensor.cc operation.cc)

# Compute owns matrix semantics; SIMD owns the one-time ISA decision and the
# locked provider/compiler machinery. Generated fixed-target TUs place the
# selected Highway body directly in hyle_compute, so no register ABI or runtime
# adapter crosses the module boundary.
hyle_add_fat_kernel(
  TARGET hyle_compute
  NAME compute_matmul
  IMPL "src/compute/matmul_kernel_impl.h"
  TARGETS scalar sse4 avx2 avx512 neon sve rvv)
