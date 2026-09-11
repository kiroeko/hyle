# Compute owns the backend-neutral device/context/tensor and prepared-operation
# contracts. The current product sources provide the synchronous CPU route;
# optional providers are added as separate implementation factors.
hyle_add_module(compute SOURCES device.cc tensor.cc operation.cc)
