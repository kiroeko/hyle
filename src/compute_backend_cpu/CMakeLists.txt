# Explicit caller-thread CPU backend. Operation kernels live in provider
# factors; this target owns only Device/Context/Queue/resource hooks.
hyle_add_module(compute_backend_cpu SOURCES cpu_backend.cc)
target_compile_definitions(hyle_compute_backend_cpu PUBLIC HYLE_COMPUTE_CPU=1)
hyle_register_package_dependency(
  NAME compute_cpu OWNER compute_backend_cpu FEATURES compute_cpu)
target_include_directories(hyle_compute_backend_cpu PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../compute")
