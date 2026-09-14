hyle_add_header_self_contained_test(compute_backend_cpu
  HEADER_DIR compute
  HEADERS backends/cpu.h)
hyle_add_test(compute_backend_cpu SOURCES
  ../compute/device_test.cc
  ../compute/host_kernel_test.cc
  ../compute/custom_kernel_test.cc
  runtime_provider_test.cc)
target_include_directories(hyle_compute_backend_cpu_test PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/compute")
