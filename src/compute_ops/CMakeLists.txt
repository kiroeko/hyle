# Provider-neutral schema implementation. Runtime views, contexts, queues and
# kernels remain in their owning factors.
hyle_audit_in_tree_dependencies(random123_counter)
hyle_add_module(compute_ops SOURCES operation.cc tensor_desc.cc low_precision.cc
  weight_decode.cc blockwise_w4a8.cc sampling_ops.cc)
hyle_register_package_dependency(
  NAME random123_counter OWNER compute_ops
  LOCKED_DEPENDENCIES random123_counter
  RUNTIME_FILE_SPECS
    "${HYLE_SOURCE_DIR}/third_party/random123_counter/LICENSE|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|random123-counter-LICENSE"
    "${HYLE_SOURCE_DIR}/third_party/random123_counter/NOTICE.md|${CMAKE_INSTALL_DATADIR}/licenses/Hyle|random123-counter-NOTICE.md")
