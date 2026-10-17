# Explicit caller-thread CPU backend. Operation kernels live in provider
# factors; this target owns only Device/Context/Queue/resource hooks.
hyle_add_module(compute_backend_cpu SOURCES cpu_backend.cc)
hyle_target_compute_capability(hyle_compute_backend_cpu cpu)
hyle_register_package_dependency(
  NAME compute_cpu OWNER compute_backend_cpu )
target_include_directories(hyle_compute_backend_cpu PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../compute")
