# Backend/provider-neutral runtime only. CPU device hooks and native kernels
# are independent factors, so other backends do not inherit host/SIMD code.
hyle_add_module(compute SOURCES
  completion.cc
  device.cc
  operation.cc
  provider.cc
  tensor_view.cc)
