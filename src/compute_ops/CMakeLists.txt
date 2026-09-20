# Provider-neutral schema implementation. Runtime views, contexts, queues and
# kernels remain in their owning factors.
hyle_add_module(compute_ops SOURCES operation.cc tensor_desc.cc low_precision.cc)
