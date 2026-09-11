# Compute v1 is a clean CPU vertical slice. Device/context/tensor contracts are
# HYLE-owned; prepared RMSNorm binds once to the existing SIMD leaf route.
hyle_add_module(compute SOURCES device.cc tensor.cc operation.cc)
