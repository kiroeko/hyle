// Included inside the locked Vulkan translation unit, after Runtime/Active.
namespace hyle::compute::ggml_vulkan::bridge {
struct Storage {
  // Upstream callbacks keep their original device context; only allocation
  // catches exceptions before returning through gallocr's C cleanup path.
  ggml_backend_buffer_type type;
  Runtime *runtime = nullptr;
  ggml_gallocr_t allocator = nullptr;
  Failure failure;
};
static_assert(offsetof(Storage, type) == 0);

namespace {
ggml_backend_buffer *allocate_graph_buffer(
    ggml_backend_buffer_type *type, std::size_t bytes) noexcept {
  auto &owner = *reinterpret_cast<Storage *>(type);
  try {
    return ggml_backend_vk_buffer_type_alloc_buffer(type, bytes);
  } catch (...) {
    owner.failure = current_failure();
    return nullptr;
  }
}
Failure check_tensor(Runtime *runtime, const ggml_tensor *tensor,
                     bool owns_imports) {
  for (const auto dimension : tensor->ne)
    if (dimension <= 0)
      return {Error::Unsupported};
  if (ggml_nbytes(tensor) > runtime->instance.devices[0]->max_buffer_size)
    return {Error::Unsupported};
  if (tensor->buffer) {
    if (!owns_imports || !ggml_backend_buffer_is_vk(tensor->buffer))
      return {Error::InvalidArgument};
    const auto *context = static_cast<const ggml_backend_vk_buffer_context *>(
        tensor->buffer->context);
    if (context->device.lock() != runtime->instance.devices[0])
      return {Error::InvalidArgument};
  } else if (tensor->data) {
    return {Error::InvalidArgument};
  }
  return {};
}
} // namespace

void destroy_storage(Storage *storage) noexcept {
  if (!storage)
    return;
  Active active(storage->runtime);
  ggml_gallocr_free(storage->allocator);
  delete storage;
}

Failure create_storage(Runtime *runtime, ggml_cgraph *graph, bool owns_imports,
                       Storage **result, std::size_t *bytes) noexcept {
  *result = nullptr;
  *bytes = 0;
  if (!graph || graph->n_nodes <= 0 || runtime->device_lost.load())
    return {Error::InvalidState};
  try {
    Active active(runtime);
    const auto native = ggml_backend_reg_dev_get(ggml_backend_vk_reg(), 0);
    bool has_output = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
      const auto *node = graph->nodes[i];
      const auto valid = check_tensor(runtime, node, owns_imports);
      if (valid.error != Error::None)
        return valid;
      if (!ggml_backend_vk_device_supports_op(native, node))
        return {Error::Unsupported};
      has_output = has_output || (node->flags & GGML_TENSOR_FLAG_OUTPUT);
    }
    if (!has_output)
      return {Error::InvalidArgument};
    for (int i = 0; i < graph->n_leafs; ++i) {
      const auto valid = check_tensor(runtime, graph->leafs[i], owns_imports);
      if (valid.error != Error::None)
        return valid;
    }
    std::unique_ptr<Storage, decltype(&destroy_storage)> storage(
        new Storage(), destroy_storage);
    storage->runtime = runtime;
    storage->type = *ggml_backend_vk_buffer_type(0);
    storage->type.iface.alloc_buffer = allocate_graph_buffer;
    storage->allocator = ggml_gallocr_new(&storage->type);
    if (!storage->allocator)
      return {Error::Exhausted};
    if (!ggml_gallocr_reserve(storage->allocator, graph) ||
        !ggml_gallocr_alloc_graph(storage->allocator, graph)) {
      const auto failure = storage->failure.error == Error::None
          ? Failure{Error::Exhausted} : storage->failure;
      if (failure.native == VK_ERROR_DEVICE_LOST)
        runtime->device_lost.store(true);
      return failure;
    }
    *bytes = ggml_gallocr_get_buffer_size(storage->allocator, 0);
    *result = storage.release();
    return {};
  } catch (...) {
    return current_failure();
  }
}
} // namespace hyle::compute::ggml_vulkan::bridge
