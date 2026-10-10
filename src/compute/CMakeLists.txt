# Backend/provider-neutral runtime only. CPU device hooks and native kernels
# are independent factors, so other backends do not inherit host/SIMD code.
hyle_add_module(compute TRACE SOURCES
  command_capture.cc
  completion.cc
  custom_kernel.cc
  device.cc
  operation.cc
  provider.cc
  resource_budget.cc
  resource_leases.cc
  route_policy.cc
  result.cc
  shape_domain.cc
  shape_program.cc
  tensor_view.cc
  transfer.cc
  workload.cc)
