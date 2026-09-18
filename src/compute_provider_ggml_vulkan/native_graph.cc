#include "native_graph.h"

#include <new>
#include <utility>

namespace hyle::compute::ggml_vulkan::detail {
struct NativeGraph::State final {
  NativeDevice device;
  MetadataArena arena;
  std::shared_ptr<const void> imports;
  bridge::Storage *storage = nullptr;
  ggml_cgraph *graph = nullptr;
  std::size_t bytes = 0;
  ~State() {
    auto active = device.activate();
    device.free_graph(storage);
    imports.reset();
  }
};

Result<NativeGraph> NativeGraph::allocate(
    NativeDevice device, MetadataArena arena, ggml_cgraph *graph,
    std::shared_ptr<const void> imports) noexcept {
  if (!device || !arena.get() || !graph)
    return std::unexpected(Status::InvalidArgument("ggml Vulkan graph owner"));
  try {
    NativeGraph result;
    result.state_ = std::make_shared<State>();
    auto &state = *result.state_;
    state.device = std::move(device);
    state.arena = std::move(arena);
    state.imports = std::move(imports);
    state.graph = graph;
    HYLE_RETURN_IF_ERROR(state.device.allocate_graph(
        graph, state.imports != nullptr, &state.storage, &state.bytes));
    return result;
  } catch (const std::bad_alloc &) {
    return std::unexpected(Status::ResourceExhausted("ggml Vulkan cold graph owner"));
  }
}
ggml_cgraph *NativeGraph::get() const noexcept {
  return state_ ? state_->graph : nullptr;
}
std::size_t NativeGraph::storage_bytes() const noexcept {
  return state_ ? state_->bytes : 0;
}
} // namespace hyle::compute::ggml_vulkan::detail
