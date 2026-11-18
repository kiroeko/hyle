// Included inside the locked Vulkan translation unit, after Runtime/Active.
namespace hyle::compute::ggml_vulkan::bridge {
// This controller owns one original native vk_buffer. Borrowed chunk
// wrappers retain it, and copy its real BDA/allocation facts, never capability
// guesses. It has no Run/Work pointer and cannot create an ownership cycle.
struct RunArena {
  std::shared_ptr<const void> device_owner;
  Runtime *runtime = nullptr;
  vk_buffer buffer;
  std::size_t required = 0;
  bool sealed = false;
  ~RunArena() { Active active(runtime); ggml_vk_destroy_buffer(buffer); }
};
struct Storage {
  // Upstream callbacks keep their original device context; only allocation
  // catches exceptions before returning through gallocr's C cleanup path.
  ggml_backend_buffer_type type;
  Runtime *runtime = nullptr;
  ggml_gallocr_t allocator = nullptr;
  Failure failure;
  struct Chunk { ggml_backend_buffer *buffer; std::size_t bytes; };
  std::vector<Chunk> chunks; // Nonowning: original gallocr owns each wrapper.
  std::shared_ptr<RunArena> arena;
  bool deferred = false;
};
static_assert(offsetof(Storage, type) == 0);

namespace {
ggml_backend_buffer *allocate_graph_buffer(
    ggml_backend_buffer_type *type, std::size_t bytes) noexcept {
  auto &owner = *reinterpret_cast<Storage *>(type);
  try {
    if (!owner.arena)
      return ggml_backend_vk_buffer_type_alloc_buffer(type, bytes);
    // Keep reserve()'s original bool failure propagation. Unlike
    // reserve_n_size(), this route does not GGML_ASSERT on a failed layout.
    auto *buffer = borrow(VK_NULL_HANDLE, bytes, owner.arena, false);
    buffer->buft = type; // Preserve original Storage.type identity for Views.
    try { owner.chunks.push_back({buffer, bytes}); }
    catch (...) { ggml_backend_buffer_free(buffer); throw; }
    return buffer;
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

namespace {
bool shareable_graph(const ggml_cgraph &graph) noexcept {
  const auto unknown_leaf = [](const ggml_tensor *tensor) {
    return tensor && tensor->op == GGML_OP_NONE && !tensor->view_src &&
        !tensor->buffer;
  };
  for (int i = 0; i < graph.n_leafs; ++i)
    if (unknown_leaf(graph.leafs[i])) return false;
  for (int i = 0; i < graph.n_nodes; ++i) {
    const auto *tensor = graph.nodes[i];
    if (unknown_leaf(tensor)) return false;
    if (!(tensor->flags & GGML_TENSOR_FLAG_OUTPUT)) continue;
    while (tensor->view_src) tensor = tensor->view_src;
    // All escaping roots must already be original external ports/owners.
    // Hadamard's unbound hinted transform and future implicit constants
    // are deliberately left on their original private allocation route.
    if (!tensor->buffer) return false;
  }
  return true;
}
} // namespace

Failure create_run_arena(Runtime *runtime, std::shared_ptr<const void> owner,
                         std::shared_ptr<RunArena> *result) noexcept {
  result->reset();
  if (!runtime || !owner) return {Error::InvalidArgument};
  try {
    auto arena = std::make_shared<RunArena>();
    arena->runtime = runtime;
    arena->device_owner = std::move(owner);
    *result = std::move(arena);
    return {};
  } catch (...) { return current_failure(); }
}
Failure seal_run_arena(Runtime *runtime,
                       const std::shared_ptr<RunArena> &arena) noexcept {
  if (!arena || arena->runtime != runtime || arena->sealed)
    return {Error::InvalidState};
  try {
    Active active(runtime);
    if (arena->required) {
      auto device = runtime->instance.devices[0];
      // Original allocation policy, charge, memory flags and actual BDA.
      arena->buffer = ggml_vk_create_buffer_device(device, arena->required);
    }
    arena->sealed = true;
    return {};
  } catch (...) {
    const auto failure = current_failure();
    if (failure.native == VK_ERROR_DEVICE_LOST) runtime->device_lost.store(true);
    return failure;
  }
}
Failure materialize_storage(Runtime *runtime, Storage *storage) noexcept {
  if (!storage || storage->runtime != runtime) return {Error::InvalidArgument};
  if (!storage->deferred) return {};
  try {
    Active active(runtime);
    if (storage->arena && (!storage->arena->sealed ||
        !storage->arena->buffer || storage->chunks.size() != 1))
      return {Error::InvalidState};
    for (const auto &chunk : storage->chunks) {
      auto *context = static_cast<ggml_backend_vk_buffer_context *>(chunk.buffer->context);
      if (storage->arena) {
        const auto &backing = storage->arena->buffer;
        if (chunk.bytes > backing->size) return {Error::InvalidState};
        auto borrowed = std::make_shared<vk_buffer_struct>();
        borrowed->device = backing->device;
        borrowed->buffer = backing->buffer;
        borrowed->device_memory = backing->device_memory;
        borrowed->memory_property_flags = backing->memory_property_flags;
        borrowed->ptr = backing->ptr;
        borrowed->bda_addr = backing->bda_addr;
        borrowed->size = chunk.bytes; // Do not enlarge the graph's range.
        borrowed->external_owner = storage->arena;
        context->dev_buffer = std::move(borrowed);
      } else {
        // A multi-chunk graph stays private. This is cold classification,
        // not fallback after a failed allocation, submit or execution.
        auto device = runtime->instance.devices[0];
        context->dev_buffer = ggml_vk_create_buffer_device(device, chunk.bytes);
      }
    }
    storage->deferred = false;
    return {};
  } catch (...) {
    const auto failure = current_failure();
    if (failure.native == VK_ERROR_DEVICE_LOST) runtime->device_lost.store(true);
    return failure;
  }
}

void destroy_storage(Storage *storage) noexcept {
  if (!storage)
    return;
  Active active(storage->runtime);
  ggml_gallocr_free(storage->allocator);
  delete storage;
}

Failure create_storage_for_run(Runtime *runtime, ggml_cgraph *graph, bool owns_imports,
    const std::shared_ptr<RunArena> &arena, Storage **result,
    std::size_t *bytes) noexcept {
  *result = nullptr;
  *bytes = 0;
  if (arena && (arena->runtime != runtime || arena->sealed))
    return {Error::InvalidState};
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
    if (arena && owns_imports && shareable_graph(*graph)) storage->arena = arena;
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
    if (storage->arena) {
      storage->deferred = !storage->chunks.empty();
      if (storage->chunks.size() == 1) {
        storage->arena->required = (std::max)(storage->arena->required,
                                             storage->chunks[0].bytes);
      } else {
        // Preserve a distinct original allocation for every graph chunk.
        // No offset rebasing or overlapping multiple live chunks at zero.
        storage->arena.reset();
        const auto failure = materialize_storage(runtime, storage.get());
        if (failure.error != Error::None) return failure;
      }
    }
    *bytes = ggml_gallocr_get_buffer_size(storage->allocator, 0);
    *result = storage.release();
    return {};
  } catch (...) {
    return current_failure();
  }
}
Failure create_storage(Runtime *runtime, ggml_cgraph *graph, bool owns_imports,
                       Storage **result, std::size_t *bytes) noexcept {
  return create_storage_for_run(runtime, graph, owns_imports, {}, result, bytes);
}
} // namespace hyle::compute::ggml_vulkan::bridge
