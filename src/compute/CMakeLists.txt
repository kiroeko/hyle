# Backend/provider-neutral runtime only. CPU device hooks and native kernels
# are independent factors, so other backends do not inherit host/SIMD code.
hyle_add_module(compute TRACE SOURCES
  batch_transfer.cc
  command_capture.cc
  completion.cc
  copy_ranges.cc
  custom_kernel.cc
  device.cc
  operation.cc
  provider.cc
  resource_budget.cc
  resource_leases.cc
  retained_completion.cc
  leased_operation.cc
  route_policy.cc
  result.cc
  shape_domain.cc
  shape_program.cc
  tensor_view.cc
  transfer.cc
  workload.cc)
