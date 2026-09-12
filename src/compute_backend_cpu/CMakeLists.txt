# Explicit caller-thread CPU backend. Operation kernels live in provider
# factors; this target owns only Device/Context/Queue/resource hooks.
hyle_add_module(compute_backend_cpu SOURCES cpu_backend.cc)
target_include_directories(hyle_compute_backend_cpu PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../compute")
